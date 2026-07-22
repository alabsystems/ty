// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! On-device semantics tests for the retained-graph CTL engine, on
//! hand-written two-slot systems small enough to verify by inspection.

use tla_gpu::{run_ctl, CtlOp, GpuCtlConfig, GpuCtlSpec};

fn cuda_available() -> bool {
    if tla_gpu::probe().is_err() {
        eprintln!("skipping CTL engine test: no usable CUDA device");
        return false;
    }
    true
}

/// Two slots; action 0 moves a token 0→1, action 1 moves it back.
/// Reachable set from [1,0]: {[1,0], [0,1]} with a 2-cycle (no deadlocks).
const TOGGLE_ACTIONS: &str = r#"
static __device__ int ty_gpu_action_0(const long long* s, long long* t) {
  if (s[0] < 1) return 0;
  t[0] = s[0] - 1; t[1] = s[1] + 1; return 1;
}
static __device__ int ty_gpu_action_1(const long long* s, long long* t) {
  if (s[1] < 1) return 0;
  t[1] = s[1] - 1; t[0] = s[0] + 1; return 1;
}
static __device__ int ty_gpu_invariants_ok(const long long* s) { return 1; }
"#;

/// One-way variant: only 0→1; [0,1] is a deadlock.
const ONE_WAY_ACTIONS: &str = r#"
static __device__ int ty_gpu_action_0(const long long* s, long long* t) {
  if (s[0] < 1) return 0;
  t[0] = s[0] - 1; t[1] = s[1] + 1; return 1;
}
static __device__ int ty_gpu_invariants_ok(const long long* s) { return 1; }
"#;

/// atom 0: slot1 >= 1; atom 1: slot0 >= 2 (unreachable); atom 2: sum == 1.
const ATOMS: &str = r#"
static __device__ int ty_gpu_atom_0(const long long* s) { return s[1] >= 1; }
static __device__ int ty_gpu_atom_1(const long long* s) { return s[0] >= 2; }
static __device__ int ty_gpu_atom_2(const long long* s) { return s[0] + s[1] == 1; }
"#;

fn spec(actions: &str, action_count: usize) -> GpuCtlSpec {
    GpuCtlSpec {
        slots: 2,
        action_count,
        actions_src: actions.to_string(),
        atoms_src: ATOMS.to_string(),
        atom_count: 3,
        init_rows: vec![1, 0],
    }
}

fn atom(k: usize) -> CtlOp {
    CtlOp::Atom(k)
}

#[test]
fn toggle_cycle_semantics() {
    if !cuda_available() {
        return;
    }
    let formulas = vec![
        CtlOp::EF(Box::new(atom(0))), // reaches [0,1]      => T
        CtlOp::EF(Box::new(atom(1))), // slot0 >= 2 never   => F
        CtlOp::ag(atom(2)),           // sum invariant      => T
        CtlOp::EG(Box::new(atom(2))), // cycle stays in sum => T
        CtlOp::EX(Box::new(atom(0))), // succ [0,1]         => T
        CtlOp::ax(atom(0)),           // only succ [0,1]    => T
        CtlOp::af(atom(0)),           // alternating path   => T
        CtlOp::EU(Box::new(atom(2)), Box::new(atom(0))), // stays in sum until => T
        CtlOp::au(atom(2), atom(0)),  // all paths          => T
        CtlOp::EG(Box::new(atom(0))), // must leave [0,1]?  => see below
    ];
    // EG(atom0) at init [1,0]: atom0 false there => F.
    let outcome = run_ctl(
        &spec(TOGGLE_ACTIONS, 2),
        &GpuCtlConfig::default(),
        &formulas,
    )
    .expect("engine should run");
    assert_eq!(outcome.distinct_states, 2);
    assert_eq!(
        outcome.verdicts,
        vec![true, false, true, true, true, true, true, true, true, false],
    );
}

/// EGF = E(GF ·), the fair-cycle / deep-LTL persistence carrier, on the toggle
/// 2-cycle {[1,0]↔[0,1]}.
#[test]
fn egf_fair_cycle_semantics() {
    if !cuda_available() {
        return;
    }
    let formulas = vec![
        CtlOp::EGF(Box::new(atom(0))), // cycle visits [0,1] i.o.        => T
        CtlOp::EGF(Box::new(atom(1))), // atom1 (slot0>=2) never         => F
        CtlOp::EGF(Box::new(atom(2))), // sum==1 i.o.                    => T
        CtlOp::afg(atom(2)),           // A(FG sum==1): all paths stay   => T
        CtlOp::afg(atom(0)),           // A(FG slot1>=1): cycle leaves   => F
    ];
    let outcome = run_ctl(
        &spec(TOGGLE_ACTIONS, 2),
        &GpuCtlConfig::default(),
        &formulas,
    )
    .expect("engine should run");
    assert_eq!(outcome.distinct_states, 2);
    assert_eq!(outcome.verdicts, vec![true, false, true, true, false]);
}

/// Deadlock-stutter soundness pin: on the one-way system [1,0]->[0,1] where
/// [0,1] is a deadlock, an infinite stutter at the deadlock IS a fair path.
/// `EGF(atom0)` must be TRUE because atom0 holds at the deadlock forever —
/// this fails without the `deadlock ∧ M` disjunct in EXˢ.
#[test]
fn egf_deadlock_stutter_is_a_fair_witness() {
    if !cuda_available() {
        return;
    }
    let one_way = GpuCtlSpec {
        slots: 2,
        action_count: 1,
        actions_src: ONE_WAY_ACTIONS.to_string(),
        atoms_src: ATOMS.to_string(),
        atom_count: 3,
        init_rows: vec![1, 0],
    };
    let formulas = vec![
        CtlOp::EGF(Box::new(atom(0))), // deadlock [0,1] holds atom0 i.o. => T
        CtlOp::EGF(Box::new(atom(2))), // sum==1 at the deadlock forever  => T
        CtlOp::EGF(Box::new(atom(1))), // atom1 never                     => F
        CtlOp::afg(atom(0)),           // A(FG slot1>=1): ends at [0,1]    => T
    ];
    let outcome =
        run_ctl(&one_way, &GpuCtlConfig::default(), &formulas).expect("engine should run");
    assert_eq!(outcome.distinct_states, 2);
    assert_eq!(outcome.verdicts, vec![true, true, false, true]);
}

/// Duplicate initial rows must dedup to one distinct initial state, and the
/// per-formula verdict must be the truth at THAT state only — never over the
/// non-initial reachable states that follow it in the arena. Regression lock
/// for the init-verdict-extraction hardening: `Not(atom0)` = "slot1 == 0" is
/// TRUE at the init [1,0] but FALSE at its successor [0,1], so a verdict that
/// wrongly read past the deduped-init boundary would return FALSE.
#[test]
fn duplicate_init_rows_verdict_is_init_only() {
    if !cuda_available() {
        return;
    }
    let dup_spec = GpuCtlSpec {
        slots: 2,
        action_count: 2,
        actions_src: TOGGLE_ACTIONS.to_string(),
        atoms_src: ATOMS.to_string(),
        atom_count: 3,
        init_rows: vec![1, 0, 1, 0], // same initial marking listed twice
    };
    let formulas = vec![
        CtlOp::Not(Box::new(atom(0))), // slot1 == 0: TRUE at init [1,0] only
        atom(0),                       // slot1 >= 1: FALSE at init [1,0]
    ];
    let outcome =
        run_ctl(&dup_spec, &GpuCtlConfig::default(), &formulas).expect("engine should run");
    // Dedup keeps one distinct initial state; the reachable set is still
    // {[1,0],[0,1]}.
    assert_eq!(outcome.distinct_states, 2);
    assert_eq!(outcome.verdicts, vec![true, false]);
}

#[test]
fn one_way_deadlock_semantics() {
    if !cuda_available() {
        return;
    }
    let formulas = vec![
        // EG(sum==1): the single maximal path [1,0],[0,1] stays in sum==1,
        // ending at a deadlock => T (maximal-path semantics).
        CtlOp::EG(Box::new(atom(2))),
        // EG(slot1==0 side): atom0 false at init => EG(atom0) at init F.
        CtlOp::EG(Box::new(atom(0))),
        // AF(atom0): the only path reaches [0,1] => T.
        CtlOp::af(atom(0)),
        // EX(true) at init: has a successor => T.
        CtlOp::EX(Box::new(CtlOp::True)),
        // AG(EF(atom0)): from [0,1] (deadlock, atom0 holds) EF(atom0)=T;
        // from [1,0] reaches it => T.
        CtlOp::ag(CtlOp::EF(Box::new(atom(0)))),
        // AG(EF(slot0>=1... via Not(atom0) at [1,0])): from the deadlock
        // [0,1] nothing reaches a Not(atom0) state => F.
        CtlOp::ag(CtlOp::EF(Box::new(CtlOp::Not(Box::new(atom(0)))))),
    ];
    let outcome = run_ctl(
        &spec(ONE_WAY_ACTIONS, 1),
        &GpuCtlConfig::default(),
        &formulas,
    )
    .expect("engine should run");
    assert_eq!(outcome.distinct_states, 2);
    assert_eq!(outcome.verdicts, vec![true, false, true, true, true, false],);
}
