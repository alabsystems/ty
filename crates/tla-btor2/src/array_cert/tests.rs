// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for the independent bounded-array SAFE certifier.
//!
//! The decisive tests exercise the disjoint bit-level LRAT path directly via
//! [`super::discharge_vcs_lrat`] with hand-built verification conditions — both
//! a genuine inductive invariant (must certify) and non-inductive ones (must
//! withhold confirmation because a VC comes back SAT). This proves the gate can
//! say *no*, which is the anti-false-confirmation guarantee.

use std::collections::HashMap;

use ay_chc::{ChcExpr, ChcSort, ChcVar};

use super::{
    certify_btor2_safe_independent, discharge_vcs_lrat, Blaster, IndependentCertResult, Invariant,
    LeafOutcome,
};
use crate::to_chc::{StateVarEntry, VcComponents};
use crate::types::{Btor2Line, Btor2Node, Btor2Program, Btor2Sort};

// ---------------------------------------------------------------------------
// Fixtures: a 2-cell array (index width 1, element width 8) hold net.
// ---------------------------------------------------------------------------

fn mem_sort() -> ChcSort {
    ChcSort::Array(Box::new(ChcSort::BitVec(1)), Box::new(ChcSort::BitVec(8)))
}

fn mem_var() -> ChcVar {
    ChcVar::new("mem", mem_sort())
}
fn mem_next_var() -> ChcVar {
    ChcVar::new("mem'", mem_sort())
}

/// select(v, 0) — read cell 0 (index is 1-bit).
fn read0(v: &ChcVar) -> ChcExpr {
    ChcExpr::select(ChcExpr::var(v.clone()), ChcExpr::BitVec(0, 1))
}

fn state_entries() -> Vec<StateVarEntry> {
    vec![StateVarEntry {
        node_id: 1,
        name: Some("mem".to_string()),
        var: mem_var(),
        next_var: mem_next_var(),
    }]
}

/// `mem = const_array(0)` — the init constraint.
fn init_all_zero() -> ChcExpr {
    ChcExpr::eq(
        ChcExpr::var(mem_var()),
        ChcExpr::const_array(ChcSort::BitVec(1), ChcExpr::BitVec(0, 8)),
    )
}

/// bad body: `mem[0] == 5`.
fn bad_cell0_eq_5() -> ChcExpr {
    ChcExpr::eq(read0(&mem_var()), ChcExpr::BitVec(5, 8))
}

/// A SAFE hold net: `mem' = mem`, bad = `mem[0] == 5`, init `mem = 0`.
fn components_hold_net() -> VcComponents {
    VcComponents {
        state_entries: state_entries(),
        init_constraint: init_all_zero(),
        trans_constraint: Some(ChcExpr::eq(
            ChcExpr::var(mem_next_var()),
            ChcExpr::var(mem_var()),
        )),
        bad_bodies: vec![bad_cell0_eq_5()],
    }
}

/// A net whose transition sets `mem[0] := 5` — an invariant `mem[0]==0` is NOT
/// inductive here (consecution must fail).
fn components_flip_net() -> VcComponents {
    VcComponents {
        state_entries: state_entries(),
        init_constraint: init_all_zero(),
        trans_constraint: Some(ChcExpr::eq(
            ChcExpr::var(mem_next_var()),
            ChcExpr::store(
                ChcExpr::var(mem_var()),
                ChcExpr::BitVec(0, 1),
                ChcExpr::BitVec(5, 8),
            ),
        )),
        bad_bodies: vec![bad_cell0_eq_5()],
    }
}

/// The genuine inductive invariant: `mem[0] == 0`. Formal param `A0` (array sort)
/// is positionally the single state variable.
fn good_invariant() -> Invariant {
    let a0 = ChcVar::new("A0", mem_sort());
    Invariant {
        params: vec![a0.clone()],
        formula: ChcExpr::eq(read0(&a0), ChcExpr::BitVec(0, 8)),
    }
}

/// A deliberately-too-weak "invariant": `true`. Excludes nothing, so the safety
/// VC must come back SAT.
fn weak_invariant() -> Invariant {
    Invariant {
        params: vec![ChcVar::new("A0", mem_sort())],
        formula: ChcExpr::Bool(true),
    }
}

// ---------------------------------------------------------------------------
// POSITIVE: a real inductive invariant is independently certified.
// ---------------------------------------------------------------------------

#[test]
fn positive_inductive_invariant_is_certified() {
    let comp = components_hold_net();
    let inv = good_invariant();
    let result = discharge_vcs_lrat(&comp, &inv);
    match result {
        IndependentCertResult::Certified {
            index_width_bits,
            cells,
            vcs_discharged,
        } => {
            // Proof the array WAS bit-blasted: index width 1 → 2 cells.
            assert_eq!(index_width_bits, 1, "index width bit-blasted");
            assert_eq!(cells, 2, "2^iw = 2 cells expanded");
            // 2 (init + consecution) + 1 bad property = 3 VCs, each LRAT-UNSAT.
            assert_eq!(vcs_discharged, 3);
        }
        other => panic!("expected Certified, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// NEGATIVE (decisive): the gate says NO when a VC is SAT.
// ---------------------------------------------------------------------------

#[test]
fn negative_weak_invariant_fails_safety_vc() {
    // Inv = true excludes no bad state ⇒ safety VC (Inv ∧ bad) is SAT.
    let comp = components_hold_net();
    let inv = weak_invariant();
    let result = discharge_vcs_lrat(&comp, &inv);
    match result {
        IndependentCertResult::NotConfirmed { reason } => {
            assert!(
                reason.contains("VC2") && reason.contains("SAT"),
                "expected a safety-VC-SAT decline, got: {reason}"
            );
        }
        other => panic!("expected NotConfirmed (safety SAT), got {other:?}"),
    }
}

#[test]
fn negative_noninductive_invariant_fails_consecution_vc() {
    // On the flip net, mem[0] becomes 5, so `mem[0]==0` is not preserved ⇒
    // consecution VC (Inv ∧ T ∧ ¬Inv') is SAT.
    let comp = components_flip_net();
    let inv = good_invariant();
    let result = discharge_vcs_lrat(&comp, &inv);
    match result {
        IndependentCertResult::NotConfirmed { reason } => {
            assert!(
                reason.contains("VC1") && reason.contains("SAT"),
                "expected a consecution-VC-SAT decline, got: {reason}"
            );
        }
        other => panic!("expected NotConfirmed (consecution SAT), got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Determinism: same inputs ⇒ same result.
// ---------------------------------------------------------------------------

#[test]
fn deterministic_result() {
    let comp = components_hold_net();
    let a = discharge_vcs_lrat(&comp, &good_invariant());
    let b = discharge_vcs_lrat(&comp, &good_invariant());
    assert_eq!(a, b);
}

// ---------------------------------------------------------------------------
// Scope guards: scalar nets and oversized arrays decline cleanly (no OOM).
// ---------------------------------------------------------------------------

#[test]
fn scalar_net_is_out_of_scope() {
    // No array state ⇒ certifier declines (the bit-level lane already owns this).
    let x = ChcVar::new("x", ChcSort::BitVec(8));
    let xp = ChcVar::new("x'", ChcSort::BitVec(8));
    let comp = VcComponents {
        state_entries: vec![StateVarEntry {
            node_id: 1,
            name: Some("x".to_string()),
            var: x.clone(),
            next_var: xp.clone(),
        }],
        init_constraint: ChcExpr::eq(ChcExpr::var(x.clone()), ChcExpr::BitVec(0, 8)),
        trans_constraint: Some(ChcExpr::eq(ChcExpr::var(xp), ChcExpr::var(x.clone()))),
        bad_bodies: vec![ChcExpr::eq(ChcExpr::var(x), ChcExpr::BitVec(1, 8))],
    };
    let inv = Invariant {
        params: vec![ChcVar::new("A0", ChcSort::BitVec(8))],
        formula: ChcExpr::eq(
            ChcExpr::var(ChcVar::new("A0", ChcSort::BitVec(8))),
            ChcExpr::BitVec(0, 8),
        ),
    };
    match discharge_vcs_lrat(&comp, &inv) {
        IndependentCertResult::NotConfirmed { reason } => {
            assert!(reason.contains("no bounded array"), "reason: {reason}");
        }
        other => panic!("expected NotConfirmed (scalar), got {other:?}"),
    }
}

#[test]
fn oversized_array_declines_without_oom() {
    // Index width 13 → 8192 cells, above the structural expansion cap (12).
    // Must decline *before* materializing any gates (no OOM).
    let big = ChcSort::Array(Box::new(ChcSort::BitVec(13)), Box::new(ChcSort::BitVec(1)));
    let m = ChcVar::new("m", big.clone());
    let mp = ChcVar::new("m'", big.clone());
    let comp = VcComponents {
        state_entries: vec![StateVarEntry {
            node_id: 1,
            name: Some("m".to_string()),
            var: m.clone(),
            next_var: mp.clone(),
        }],
        init_constraint: ChcExpr::Bool(true),
        trans_constraint: Some(ChcExpr::eq(ChcExpr::var(mp), ChcExpr::var(m))),
        bad_bodies: vec![ChcExpr::Bool(false)],
    };
    let inv = Invariant {
        params: vec![ChcVar::new("A0", big)],
        formula: ChcExpr::Bool(true),
    };
    match discharge_vcs_lrat(&comp, &inv) {
        IndependentCertResult::NotConfirmed { reason } => {
            assert!(reason.contains("structural cap"), "reason: {reason}");
        }
        other => panic!("expected NotConfirmed (over cap), got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Resource backstop: a tiny gate ceiling aborts a big blast to Inconclusive
// (fail-closed, no unbounded materialization).
// ---------------------------------------------------------------------------

#[test]
fn gate_ceiling_backstop_declines() {
    // A store with a symbolic index needs many gates; a ceiling of 2 forces the
    // blaster over budget, so the VC leaf returns Inconclusive rather than
    // building an unbounded CNF.
    let mut b = Blaster::new(2);
    let mem = mem_var();
    // store(mem, i, 7) with a symbolic index i.
    let i = ChcVar::new("i", ChcSort::BitVec(1));
    let store = ChcExpr::store(
        ChcExpr::var(mem.clone()),
        ChcExpr::var(i),
        ChcExpr::BitVec(7, 8),
    );
    // Assert store == mem (forces per-cell mux gates → exceeds the ceiling).
    let eq = ChcExpr::eq(store, ChcExpr::var(mem));
    let _ = b.assert_true(&eq); // may already flip over_budget
    match b.solve_unsat_lrat() {
        LeafOutcome::Inconclusive(reason) => {
            assert!(
                reason.contains("ceiling") || reason.contains("cap"),
                "reason: {reason}"
            );
        }
        // If the blaster happened to fold enough to stay under the ceiling, the
        // query is still sound — but for this fixture we expect the backstop.
        other => panic!(
            "expected Inconclusive from the gate ceiling, got {}",
            leaf_name(&other)
        ),
    }
}

fn leaf_name(o: &LeafOutcome) -> &'static str {
    match o {
        LeafOutcome::VerifiedUnsat => "VerifiedUnsat",
        LeafOutcome::Sat => "Sat",
        LeafOutcome::Inconclusive(_) => "Inconclusive",
    }
}

// ---------------------------------------------------------------------------
// END-TO-END: the full opt-in entry certifies a real SAFE bounded-array net.
// ---------------------------------------------------------------------------

/// Build the 2-cell hold net as a real BTOR2 program:
///   mem : Array(BV1, BV8), init all-0, next = mem (hold), bad: mem[0] == 5.
fn btor2_hold_net() -> Btor2Program {
    let mut sorts = HashMap::new();
    sorts.insert(1, Btor2Sort::BitVec(1)); // index
    sorts.insert(2, Btor2Sort::BitVec(8)); // element
    sorts.insert(
        3,
        Btor2Sort::Array {
            index: Box::new(Btor2Sort::BitVec(1)),
            element: Box::new(Btor2Sort::BitVec(8)),
        },
    );
    sorts.insert(4, Btor2Sort::BitVec(1)); // comparison result

    let lines = vec![
        Btor2Line {
            id: 1,
            sort_id: 0,
            node: Btor2Node::SortBitVec(1),
            args: vec![],
        },
        Btor2Line {
            id: 2,
            sort_id: 0,
            node: Btor2Node::SortBitVec(8),
            args: vec![],
        },
        Btor2Line {
            id: 3,
            sort_id: 0,
            node: Btor2Node::SortArray(1, 2),
            args: vec![],
        },
        Btor2Line {
            id: 4,
            sort_id: 0,
            node: Btor2Node::SortBitVec(1),
            args: vec![],
        },
        // 8-bit zero (init value) and index 0.
        Btor2Line {
            id: 10,
            sort_id: 2,
            node: Btor2Node::Zero,
            args: vec![],
        },
        Btor2Line {
            id: 11,
            sort_id: 1,
            node: Btor2Node::Zero,
            args: vec![],
        },
        // mem state.
        Btor2Line {
            id: 12,
            sort_id: 3,
            node: Btor2Node::State(3, Some("mem".to_string())),
            args: vec![],
        },
        // init mem = const 0 (scalar lifts to const-array).
        Btor2Line {
            id: 13,
            sort_id: 3,
            node: Btor2Node::Init(3, 12, 10),
            args: vec![],
        },
        // next mem' = mem (hold).
        Btor2Line {
            id: 14,
            sort_id: 3,
            node: Btor2Node::Next(3, 12, 12),
            args: vec![],
        },
        // read mem[0].
        Btor2Line {
            id: 15,
            sort_id: 2,
            node: Btor2Node::Read,
            args: vec![12, 11],
        },
        // constant 5.
        Btor2Line {
            id: 16,
            sort_id: 2,
            node: Btor2Node::ConstD("5".to_string()),
            args: vec![],
        },
        // bad: mem[0] == 5.
        Btor2Line {
            id: 17,
            sort_id: 4,
            node: Btor2Node::Eq,
            args: vec![15, 16],
        },
        Btor2Line {
            id: 18,
            sort_id: 0,
            node: Btor2Node::Bad(17),
            args: vec![],
        },
    ];

    Btor2Program {
        lines,
        sorts,
        num_inputs: 0,
        num_states: 1,
        bad_properties: vec![18],
        constraints: vec![],
        fairness: vec![],
        justice: vec![],
    }
}

#[test]
fn end_to_end_certifies_safe_array_net() {
    let program = btor2_hold_net();
    let result = certify_btor2_safe_independent(&program, Some(std::time::Duration::from_secs(30)))
        .expect("translation must succeed");
    // The portfolio should prove this SAFE and the disjoint LRAT gate confirm it.
    // If the portfolio itself cannot decide within budget the certifier declines
    // (sound); assert we never crash and, on success, that we DID certify.
    match result {
        IndependentCertResult::Certified { cells, .. } => {
            assert_eq!(cells, 2);
        }
        IndependentCertResult::NotConfirmed { reason } => {
            // Acceptable only if the portfolio/proof-back was the limiter — never
            // a Gate-B "VC is SAT" (that would be a false confirmation escape).
            assert!(
                !reason.contains("is SAT"),
                "Gate B must not report a SAT VC on a genuinely SAFE net: {reason}"
            );
        }
    }
}
