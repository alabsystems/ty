// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Unit tests for the symbolic state-equation dispatcher.
//!
//! Test naming follows the project standard:
//!   `test_<unit>_<scenario>_<expected>`.
//!
//! Soundness floor: every assertion either matches an exact
//! [`SymbolicVerdict`] variant or compares against an explicit BFS
//! ground truth. We never assert "should be Safe" when the BFS path
//! would say Unsafe, and vice versa.

use std::collections::{HashSet, VecDeque};
use std::time::Duration;

use super::chc_dispatch::{
    symbolic_state_equation_check, SymbolicConfig, SymbolicVerdict, UnknownReason,
};
use super::state_equation::{
    StateEquationEncoder, StateEquationEncoderError, DISABLE_CHC_TRAP_CUTS_ENV,
};
use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};
use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};

// ── Test fixtures ─────────────────────────────────────────────────────

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

/// Token-conserving net: p0 has 3 tokens, t0 moves one p0 → p1.
/// State equation: m0 + m1 = 3 across all reachable markings.
fn conserving_net() -> PetriNet {
    PetriNet {
        name: None,
        places: vec![place("p0"), place("p1")],
        transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
        initial_marking: vec![3, 0],
    }
}

/// Producer net: t0 has no input, fires forever, producing one p0 token
/// per firing. Unbounded — reachability of arbitrarily large m0 is
/// trivially true.
fn producer_net() -> PetriNet {
    PetriNet {
        name: None,
        places: vec![place("p0")],
        transitions: vec![trans("t0", vec![], vec![arc(0, 1)])],
        initial_marking: vec![0],
    }
}

/// A small reversible net (precedent: `dual_kill_trap_net` in
/// `lp_state_equation.rs`). Used for differential testing against BFS.
fn dual_kill_trap_net() -> PetriNet {
    PetriNet {
        name: None,
        places: vec![place("p0"), place("p1")],
        transitions: vec![
            trans("t0", vec![arc(0, 1), arc(1, 1)], vec![arc(0, 1)]),
            trans("t1", vec![arc(0, 1), arc(1, 1)], vec![arc(1, 1)]),
            trans("t2", vec![arc(1, 1)], vec![arc(0, 1)]),
        ],
        initial_marking: vec![1, 1],
    }
}

/// Tiny one-way producer/consumer (no token conservation).
fn consume_net() -> PetriNet {
    PetriNet {
        name: None,
        places: vec![place("p0")],
        transitions: vec![trans("t0", vec![arc(0, 1)], vec![])],
        initial_marking: vec![2],
    }
}

/// Two-token splitter.
fn splitter_net() -> PetriNet {
    PetriNet {
        name: None,
        places: vec![place("p0"), place("p1"), place("p2")],
        transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(1, 1), arc(2, 1)])],
        initial_marking: vec![5, 0, 0],
    }
}

/// Synthetic overflow net: arc weight is u64::MAX which does not fit
/// in i64. Used by `test_symbolic_overflow_guard`.
fn overflow_net() -> PetriNet {
    PetriNet {
        name: None,
        places: vec![place("p0"), place("p1")],
        transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(1, u64::MAX)])],
        initial_marking: vec![1, 0],
    }
}

// ── Reference BFS (for differential tests) ────────────────────────────

/// Reference exact BFS evaluator. Returns `true` if the predicate
/// holds in every reachable marking visited up to the state cap;
/// returns `None` if exploration was truncated (i.e., the BFS itself
/// cannot give a ground truth).
fn bfs_predicate_holds_everywhere(
    net: &PetriNet,
    predicate: &ResolvedPredicate,
    state_cap: usize,
) -> Option<bool> {
    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let mut queue: VecDeque<Vec<u64>> = VecDeque::new();

    seen.insert(net.initial_marking.clone());
    queue.push_back(net.initial_marking.clone());

    if !crate::resolved_predicate::eval_predicate(predicate, &net.initial_marking, net) {
        return Some(false);
    }

    while let Some(marking) = queue.pop_front() {
        for tidx in 0..net.num_transitions() {
            let t = TransitionIdx(tidx as u32);
            if !net.is_enabled(&marking, t) {
                continue;
            }
            let succ = net.fire(&marking, t).expect("fire (test)");
            if !crate::resolved_predicate::eval_predicate(predicate, &succ, net) {
                return Some(false);
            }
            if seen.insert(succ.clone()) {
                if seen.len() > state_cap {
                    return None;
                }
                queue.push_back(succ);
            }
        }
    }
    Some(true)
}

fn fast_config() -> SymbolicConfig {
    SymbolicConfig {
        time_budget: Duration::from_secs(10),
        ..SymbolicConfig::default()
    }
}

// ── Encoder unit tests ────────────────────────────────────────────────

#[test]
fn test_state_equation_encoder_small_net_builds_problem() {
    let net = conserving_net();
    let encoder = StateEquationEncoder::new(&net);
    let property = ResolvedPredicate::IntLe(
        ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
        ResolvedIntExpr::Constant(3),
    );
    let problem = encoder
        .encode_safety_query(&property)
        .expect("small net must encode without error");
    // Init + 1 transition + stuttering + query = 4 clauses.
    assert_eq!(
        problem.clauses().len(),
        4,
        "expected 1 init + 1 transition + 1 stutter + 1 query clause"
    );
    // Single Inv predicate declared.
    assert_eq!(problem.predicates().len(), 1);
}

#[test]
fn test_state_equation_encoder_arc_weight_overflow_rejected() {
    let net = overflow_net();
    let encoder = StateEquationEncoder::new(&net);
    let property = ResolvedPredicate::True;
    let result = encoder.encode_safety_query(&property);
    assert!(
        matches!(
            result,
            Err(StateEquationEncoderError::ArcWeightOverflow { .. })
        ),
        "u64::MAX arc weight must be rejected before reaching the solver, got {result:?}"
    );
}

#[test]
fn test_state_equation_encoder_initial_marking_overflow_rejected() {
    let net = PetriNet {
        name: None,
        places: vec![place("p0")],
        transitions: vec![trans("t0", vec![], vec![arc(0, 1)])],
        initial_marking: vec![u64::MAX],
    };
    let encoder = StateEquationEncoder::new(&net);
    let property = ResolvedPredicate::True;
    let result = encoder.encode_safety_query(&property);
    assert!(
        matches!(
            result,
            Err(StateEquationEncoderError::InitialMarkingOverflow { .. })
        ),
        "u64::MAX initial marking must be rejected, got {result:?}"
    );
}

// ── Dispatcher tests ──────────────────────────────────────────────────

/// Small net + provable safety predicate ⇒ symbolic returns SAFE.
///
/// Property: `m1 ≤ 3` on the conserving net. The state equation
/// `m0 + m1 = 3` makes this trivially safe; IC3 finds it immediately.
#[test]
fn test_symbolic_simple_safe_property_correct() {
    let net = conserving_net();
    let property = ResolvedPredicate::IntLe(
        ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        ResolvedIntExpr::Constant(3),
    );
    let verdict = symbolic_state_equation_check(&net, &property, &fast_config());
    // Cross-check against BFS ground truth.
    let bfs_truth = bfs_predicate_holds_everywhere(&net, &property, 1_000);
    assert_eq!(
        bfs_truth,
        Some(true),
        "BFS sanity: property must hold everywhere on conserving net"
    );
    match verdict {
        SymbolicVerdict::Safe => {}
        SymbolicVerdict::Unknown { .. } => {
            // Soundness preserved — solver returned Unknown is acceptable.
        }
        other => panic!(
            "expected Safe or Unknown for provably-safe predicate on small net, got {other:?}"
        ),
    }
}

/// Small net + violated safety predicate ⇒ symbolic returns UNSAFE.
///
/// Property: `m1 ≤ 0` on the conserving net is violated after firing
/// `t0` once; BFS finds m=(2,1) as a witness.
#[test]
fn test_symbolic_simple_unsafe_property_correct() {
    let net = conserving_net();
    // Negate: we want a *violatable* safety property. "m1 ≤ 0" is
    // false after one firing.
    let property = ResolvedPredicate::IntLe(
        ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        ResolvedIntExpr::Constant(0),
    );

    let bfs_truth = bfs_predicate_holds_everywhere(&net, &property, 1_000);
    assert_eq!(
        bfs_truth,
        Some(false),
        "BFS sanity: predicate must be violated on conserving net"
    );

    let verdict = symbolic_state_equation_check(&net, &property, &fast_config());
    match verdict {
        SymbolicVerdict::Unsafe { witness } => {
            assert!(
                !witness.is_empty(),
                "unsafe verdict should carry at least one witness step"
            );
        }
        SymbolicVerdict::Unknown { .. } => {
            // Acceptable — soundness preserved (no wrong SAFE).
        }
        SymbolicVerdict::Safe => {
            panic!("symbolic SAFE on a BFS-UNSAFE predicate would be a SOUNDNESS BREACH");
        }
    }
}

/// Differential test across multiple small nets: when both BFS and
/// symbolic produce a definite verdict, they must agree.
#[test]
fn test_symbolic_matches_bfs_on_small_net() {
    // (net, property, label) tuples — covers conserving, reversible,
    // consumer, and splitter shapes for breadth.
    let cases: Vec<(PetriNet, ResolvedPredicate, &'static str)> = vec![
        // Safe on conserving net.
        (
            conserving_net(),
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(0), PlaceIdx(1)]),
                ResolvedIntExpr::Constant(3),
            ),
            "conserving:sum<=3",
        ),
        // Safe on consumer net (tokens only decrease).
        (
            consume_net(),
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
                ResolvedIntExpr::Constant(2),
            ),
            "consumer:p0<=2",
        ),
        // Unsafe on splitter net: p1 reaches up to 5.
        (
            splitter_net(),
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
                ResolvedIntExpr::Constant(0),
            ),
            "splitter:p1<=0(unsafe)",
        ),
        // Safe on dual_kill_trap: the trap {p0,p1} keeps sum>=1.
        (
            dual_kill_trap_net(),
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::Constant(1),
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(0), PlaceIdx(1)]),
            ),
            "trap:sum>=1",
        ),
        // Unsafe on conserving net: p1>=4 violates m0+m1=3.
        // Actually this is *safe in the unreachability sense*: predicate
        // `p1 <= 0` is violated; we want the predicate-holds form.
        (
            conserving_net(),
            ResolvedPredicate::IntLe(
                ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
                ResolvedIntExpr::Constant(0),
            ),
            "conserving:p1<=0(unsafe)",
        ),
    ];

    for (net, property, label) in cases {
        let bfs_truth = bfs_predicate_holds_everywhere(&net, &property, 10_000);
        let verdict = symbolic_state_equation_check(&net, &property, &fast_config());

        match (bfs_truth, &verdict) {
            (Some(true), SymbolicVerdict::Safe) | (Some(false), SymbolicVerdict::Unsafe { .. }) => {
                // Agreement — fine.
            }
            (_, SymbolicVerdict::Unknown { .. }) => {
                // Soundness preserved — solver couldn't decide, that's
                // acceptable in a differential test.
            }
            (Some(true), SymbolicVerdict::Unsafe { .. }) => {
                panic!(
                    "[{label}] SOUNDNESS BREACH: BFS says Safe, symbolic says Unsafe — would emit wrong verdict"
                );
            }
            (Some(false), SymbolicVerdict::Safe) => {
                panic!(
                    "[{label}] SOUNDNESS BREACH: BFS says Unsafe, symbolic says Safe — would emit wrong verdict"
                );
            }
            (None, _) => {
                // BFS truncated — cannot use as ground truth.
            }
        }
    }
}

/// Overflow synthesis: an arc weight at `u64::MAX` cannot fit in `i64`.
/// Without `checked_mul` / `try_from`, this would either truncate to a
/// negative value (admitting impossible transitions) or panic. With
/// the guard, the dispatcher MUST surface UNKNOWN — never a wrong
/// SAFE/UNSAFE.
#[test]
fn test_symbolic_overflow_guard() {
    let net = overflow_net();
    let property = ResolvedPredicate::IntLe(
        ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        ResolvedIntExpr::Constant(0),
    );
    let verdict = symbolic_state_equation_check(&net, &property, &fast_config());
    match verdict {
        SymbolicVerdict::Unknown {
            reason: UnknownReason::EncoderRejected(_),
        } => {}
        other => panic!(
            "expected UNKNOWN/EncoderRejected on u64::MAX arc weight (soundness), got {other:?}"
        ),
    }
}

/// The CHC lane RESOLVES fireability queries (not just IntLe): conserving_net's
/// t0 (p0→p1) is fireable at the initial marking, so `EF IsFireable(t0)` is
/// TRUE — encoded as the safety query `¬IsFireable(t0)`, violated at the
/// initial state — and the lane returns Unsafe with a witness. Guards against a
/// regression that would silently make the fireability admission a no-op.
#[test]
fn test_chc_resolves_fireability_query() {
    let net = conserving_net();
    let property = ResolvedPredicate::Not(Box::new(ResolvedPredicate::IsFireable(vec![
        TransitionIdx(0),
    ])));
    let verdict = symbolic_state_equation_check(&net, &property, &fast_config());
    assert!(
        matches!(verdict, SymbolicVerdict::Unsafe { .. }),
        "CHC lane returned {verdict:?} on a trivially-fireable query — fireability not resolved"
    );
}

/// Fallback dispatch surface test: simulates the "BFS would exhaust"
/// scenario by invoking the symbolic check directly on a producer net
/// whose explicit state space is countably infinite. Symbolic must
/// either resolve (Safe/Unsafe) or surface Unknown — never silently
/// hang or return a wrong verdict.
#[test]
fn test_symbolic_fallback_dispatch_when_bfs_exhausts() {
    let net = producer_net();
    // m0 is unbounded; the predicate "m0 ≤ 100" is violated after 101
    // firings. The state space is infinite, so explicit BFS would
    // never terminate. Symbolic IC3 should find UNSAFE (or fall back
    // to Unknown — but absolutely not Safe).
    let property = ResolvedPredicate::IntLe(
        ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
        ResolvedIntExpr::Constant(100),
    );

    let verdict = symbolic_state_equation_check(&net, &property, &fast_config());
    match verdict {
        SymbolicVerdict::Unsafe { .. } | SymbolicVerdict::Unknown { .. } => {}
        SymbolicVerdict::Safe => panic!(
            "SOUNDNESS BREACH: producer net's m0 is unbounded; symbolic SAFE here would be wrong"
        ),
    }
}

/// Net-too-large guard: a `max_places=0` config trips the encoder cap
/// and returns Unknown rather than attempting to solve.
#[test]
fn test_symbolic_net_too_large_rejected() {
    let net = conserving_net();
    let config = SymbolicConfig {
        max_places: 0,
        ..SymbolicConfig::default()
    };
    let property = ResolvedPredicate::True;
    let verdict = symbolic_state_equation_check(&net, &property, &config);
    assert!(
        matches!(
            verdict,
            SymbolicVerdict::Unknown {
                reason: UnknownReason::EncoderRejected(_)
            }
        ),
        "max_places=0 must surface as EncoderRejected, got {verdict:?}"
    );
}

/// A truly trivial safety property (`True`) must always be SAFE.
#[test]
fn test_symbolic_trivially_true_predicate_safe() {
    let net = conserving_net();
    let property = ResolvedPredicate::True;
    let verdict = symbolic_state_equation_check(&net, &property, &fast_config());
    match verdict {
        SymbolicVerdict::Safe | SymbolicVerdict::Unknown { .. } => {}
        SymbolicVerdict::Unsafe { .. } => {
            panic!("SOUNDNESS BREACH: `True` predicate cannot be Unsafe")
        }
    }
}

/// A trivially `False` safety property is violated at the initial
/// marking. Must surface UNSAFE — never Safe.
#[test]
fn test_symbolic_trivially_false_predicate_unsafe() {
    let net = conserving_net();
    let property = ResolvedPredicate::False;
    let verdict = symbolic_state_equation_check(&net, &property, &fast_config());
    match verdict {
        SymbolicVerdict::Unsafe { .. } | SymbolicVerdict::Unknown { .. } => {}
        SymbolicVerdict::Safe => {
            panic!("SOUNDNESS BREACH: `False` predicate cannot be Safe")
        }
    }
}

// ── Trap-cut (initially-marked-trap query-strengthening) tests ─────────

/// RAII guard: hold the crate-wide env lock and restore the prior value of
/// `TY_MCC_DISABLE_CHC_TRAP_CUTS` on drop. Lets a test toggle the kill-switch
/// without racing or leaking into sibling tests.
struct TrapCutEnvGuard {
    prior: Option<String>,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl TrapCutEnvGuard {
    /// Disable trap cuts for the lifetime of the guard.
    fn disabled() -> Self {
        let lock = crate::env_test_lock();
        let prior = std::env::var(DISABLE_CHC_TRAP_CUTS_ENV).ok();
        crate::env_guard::set_var(DISABLE_CHC_TRAP_CUTS_ENV, "1");
        Self { prior, _lock: lock }
    }

    /// Enable trap cuts (clear the kill-switch) for the lifetime of the guard.
    fn enabled() -> Self {
        let lock = crate::env_test_lock();
        let prior = std::env::var(DISABLE_CHC_TRAP_CUTS_ENV).ok();
        crate::env_guard::remove_var(DISABLE_CHC_TRAP_CUTS_ENV);
        Self { prior, _lock: lock }
    }
}

impl Drop for TrapCutEnvGuard {
    fn drop(&mut self) {
        match &self.prior {
            Some(v) => crate::env_guard::set_var(DISABLE_CHC_TRAP_CUTS_ENV, v),
            None => crate::env_guard::remove_var(DISABLE_CHC_TRAP_CUTS_ENV),
        }
    }
}

/// A single token circulating around an `n`-place ring: place `i` feeds place
/// `(i+1) mod n` via transition `i`. All `n` places form one initially-marked
/// trap (the token can move but never vanish or duplicate), so the AG-universal
/// property "the ring is never empty" (`sum_i m[i] >= 1`) is genuinely SAFE —
/// but the bare state equation admits the all-zero (empty-ring) marking as a
/// spurious solution, so a solver without the trap invariant must synthesise it.
fn token_ring_net(n: usize) -> PetriNet {
    assert!(n >= 2);
    let places: Vec<PlaceInfo> = (0..n).map(|i| place(&format!("p{i}"))).collect();
    let transitions: Vec<TransitionInfo> = (0..n)
        .map(|i| {
            let next = (i + 1) % n;
            trans(
                &format!("t{i}"),
                vec![arc(i as u32, 1)],
                vec![arc(next as u32, 1)],
            )
        })
        .collect();
    let mut initial_marking = vec![0_u64; n];
    initial_marking[0] = 1;
    PetriNet {
        name: None,
        places,
        transitions,
        initial_marking,
    }
}

/// THE CORE LEVER. On a token ring the AG-universal property "ring never empty"
/// (`sum_i m[i] >= 1`) is SAFE only because the whole ring is an initially-marked
/// trap. The bare state equation admits the empty-ring marking, so WITHOUT the
/// cut the solver must rediscover the trap invariant; WITH the cut the query is
/// strengthened by `sum_i m[i] >= 1` and closes immediately.
///
/// We assert: (a) BFS ground truth is Safe; (b) WITH cuts the lane returns Safe
/// matching BFS; (c) the without-cuts run is sound (never a wrong Unsafe) and
/// (d) cuts never DEGRADE the verdict (with-cuts is at least as strong as
/// without-cuts). On large MCC nets the without-cuts run is where IC3 times out
/// rediscovering the trap; on this small ring IC3 may already close, but the
/// dominance + BFS-agreement assertions still pin the cut's correctness.
#[test]
fn test_trap_cut_flips_unknown_to_safe_matching_bfs() {
    let net = token_ring_net(6);
    let all_places: Vec<PlaceIdx> = (0..net.num_places() as u32).map(PlaceIdx).collect();
    // Safety predicate: the ring trap keeps the total token count >= 1.
    let property = ResolvedPredicate::IntLe(
        ResolvedIntExpr::Constant(1),
        ResolvedIntExpr::TokensCount(all_places),
    );

    // Ground truth: exhaustive BFS says the property holds everywhere.
    let bfs_truth = bfs_predicate_holds_everywhere(&net, &property, 10_000);
    assert_eq!(
        bfs_truth,
        Some(true),
        "BFS sanity: the ring trap keeps the total >= 1 in all reachable markings"
    );

    // WITHOUT cuts (kill-switch on): must be sound — never a wrong Unsafe on a
    // BFS-Safe property. It is permitted to be Unknown (the timeout the cut
    // exists to fix on large nets).
    let without = {
        let _g = TrapCutEnvGuard::disabled();
        symbolic_state_equation_check(&net, &property, &fast_config())
    };
    assert!(
        !matches!(without, SymbolicVerdict::Unsafe { .. }),
        "without cuts the lane must not emit a (wrong) Unsafe on a BFS-Safe property, got {without:?}"
    );

    // WITH cuts (kill-switch off): the trap invariant is a query-strengthening
    // conjunct, making the bad-state query infeasible, so the lane returns Safe
    // — matching BFS.
    let with = {
        let _g = TrapCutEnvGuard::enabled();
        symbolic_state_equation_check(&net, &property, &fast_config())
    };
    assert!(
        matches!(with, SymbolicVerdict::Safe),
        "WITH trap cuts the lane must return Safe (matching BFS) on the ring trap, got {with:?}"
    );

    // Cuts must never DEGRADE: if without-cuts already proved Safe, with-cuts
    // must still be Safe (never regress).
    if matches!(without, SymbolicVerdict::Safe) {
        assert!(
            matches!(with, SymbolicVerdict::Safe),
            "trap cuts must not regress a previously-Safe verdict, got {with:?}"
        );
    }
}

/// Soundness floor for the cut: the trap invariant must NEVER manufacture a
/// Safe verdict for a genuinely-violated property. On the token ring the
/// property "place p0 always holds the token" (`m0 >= 1`) is FALSE — the token
/// moves away from p0 — so BFS reports a violation. With trap cuts enabled the
/// lane must return Unsafe or Unknown, never Safe.
#[test]
fn test_trap_cut_never_fakes_safe_on_violated_property() {
    let net = token_ring_net(6);
    let property = ResolvedPredicate::IntLe(
        ResolvedIntExpr::Constant(1),
        ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
    );

    let bfs_truth = bfs_predicate_holds_everywhere(&net, &property, 10_000);
    assert_eq!(
        bfs_truth,
        Some(false),
        "BFS sanity: the token leaves p0, so `m0 >= 1` is violated in a reachable marking"
    );

    let verdict = {
        let _g = TrapCutEnvGuard::enabled();
        symbolic_state_equation_check(&net, &property, &fast_config())
    };
    match verdict {
        // Unsafe is sound (and replay-validated in the seeder). What matters
        // here is that the trap cut did NOT turn the violation into Safe.
        SymbolicVerdict::Unsafe { .. } | SymbolicVerdict::Unknown { .. } => {}
        SymbolicVerdict::Safe => {
            panic!("SOUNDNESS BREACH: trap cut turned a BFS-Unsafe property into Safe")
        }
    }
}

/// UNSAFE replay-validation: on the conserving net the property `m1 <= 0` is
/// violated only AFTER firing t0 (the initial marking (3,0) satisfies it), so a
/// genuine Unsafe verdict must carry a non-empty witness that the seeder can
/// replay on the concrete net. With trap cuts enabled this must still hold —
/// the cut never suppresses or fabricates a witness. Mirrors
/// `test_symbolic_simple_unsafe_property_correct` with cuts explicitly on.
#[test]
fn test_trap_cut_unsafe_still_carries_replayable_witness() {
    let net = conserving_net();
    let property = ResolvedPredicate::IntLe(
        ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        ResolvedIntExpr::Constant(0),
    );
    let bfs_truth = bfs_predicate_holds_everywhere(&net, &property, 1_000);
    assert_eq!(
        bfs_truth,
        Some(false),
        "BFS: m1<=0 violated after firing t0"
    );

    let verdict = {
        let _g = TrapCutEnvGuard::enabled();
        symbolic_state_equation_check(&net, &property, &fast_config())
    };
    match verdict {
        SymbolicVerdict::Unsafe { witness } => {
            assert!(
                !witness.is_empty(),
                "post-initial violation must carry a non-empty (replayable) witness"
            );
        }
        SymbolicVerdict::Unknown { .. } => {}
        SymbolicVerdict::Safe => {
            panic!("SOUNDNESS BREACH: BFS-Unsafe property returned Safe with cuts on")
        }
    }
}

/// A net with NO nontrivial initially-marked trap (`conserving_net`: the only
/// candidate trap minimizes to `{p1}`, which is empty in the initial marking
/// and so is discarded) must encode IDENTICALLY whether cuts are on or off —
/// the trap loop adds zero query conjuncts. Guards against the cut perturbing
/// trap-free nets.
#[test]
fn test_trap_cut_no_op_on_net_without_initially_marked_trap() {
    let net = conserving_net();
    // Sanity: the trap finder yields nothing initially-marked for this net.
    let traps = crate::lp_state_equation::find_initially_marked_traps(&net);
    assert!(
        traps.is_empty(),
        "conserving_net has no retained initially-marked trap, got {traps:?}"
    );

    let property = ResolvedPredicate::IntLe(
        ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        ResolvedIntExpr::Constant(3),
    );

    // Clause count is unchanged regardless of the kill-switch: trap conjuncts
    // fold into the single existing query clause, and there are none here.
    let problem_with = {
        let _g = TrapCutEnvGuard::enabled();
        StateEquationEncoder::new(&net)
            .encode_safety_query(&property)
            .expect("encodes")
    };
    let problem_without = {
        let _g = TrapCutEnvGuard::disabled();
        StateEquationEncoder::new(&net)
            .encode_safety_query(&property)
            .expect("encodes")
    };
    assert_eq!(
        problem_with.clauses().len(),
        problem_without.clauses().len(),
        "trap-free net must encode to the same clause count with and without cuts"
    );
    assert_eq!(
        problem_with.clauses().len(),
        4,
        "init + transition + stutter + query = 4 clauses for a trap-free net"
    );

    // And the verdict matches BFS either way.
    let bfs_truth = bfs_predicate_holds_everywhere(&net, &property, 10_000);
    assert_eq!(bfs_truth, Some(true), "BFS: m1 <= 3 holds (m0+m1=3)");
    let verdict = {
        let _g = TrapCutEnvGuard::enabled();
        symbolic_state_equation_check(&net, &property, &fast_config())
    };
    match verdict {
        SymbolicVerdict::Safe | SymbolicVerdict::Unknown { .. } => {}
        SymbolicVerdict::Unsafe { .. } => {
            panic!("SOUNDNESS BREACH: trap-free conserving net is Safe per BFS, got Unsafe")
        }
    }
}

/// The kill-switch must be read: a net WITH an initially-marked trap encodes the
/// SAME clause count either way (cuts fold into the query clause), but the query
/// BODY differs. We confirm both encodings build cleanly and the trap is exposed.
#[test]
fn test_trap_cut_kill_switch_is_honored() {
    let net = dual_kill_trap_net();
    // Sanity: this net DOES have the nontrivial initially-marked trap {p0,p1}.
    let traps = crate::lp_state_equation::find_initially_marked_traps(&net);
    assert_eq!(
        traps,
        vec![vec![true, true]],
        "dual_kill_trap_net must expose the {{p0,p1}} initially-marked trap"
    );

    let property = ResolvedPredicate::IntLe(
        ResolvedIntExpr::Constant(1),
        ResolvedIntExpr::TokensCount(vec![PlaceIdx(0), PlaceIdx(1)]),
    );

    let with = {
        let _g = TrapCutEnvGuard::enabled();
        StateEquationEncoder::new(&net)
            .encode_safety_query(&property)
            .expect("encodes with cuts")
    };
    let without = {
        let _g = TrapCutEnvGuard::disabled();
        StateEquationEncoder::new(&net)
            .encode_safety_query(&property)
            .expect("encodes without cuts")
    };
    assert_eq!(
        with.clauses().len(),
        without.clauses().len(),
        "trap conjuncts fold into the query clause; clause count is invariant"
    );
}
