// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Integration tests for the public verdict + translation entry points that
//! were not otherwise exercised:
//!
//! - [`check_btor2_adaptive`] — the documented preferred (memoized, DAG-aware)
//!   solve entry point. Covers its empty-property short-circuit, its SAFE
//!   (`Unsat`) verdict on a trivially-safe model, and its UNSAFE (`Sat`)
//!   counterexample on a shallow reachable bug.
//! - [`translate_to_chc`] — the public CHC lowering. Covers the standard
//!   3-clause structure, the per-`bad`-property index contract for several
//!   properties, and its documented `# Errors` paths (over-wide bitvector,
//!   undefined node reference).
//! - [`check_btor2`] — the simple translator's empty-property contract.
//!
//! These programs are deterministic and small (shallow bug at depth <= 3, or a
//! one-step safe model) so the solver decides them quickly.

use std::collections::HashMap;
use std::time::Duration;

use tla_btor2::error::MAX_BV_WIDTH;
use tla_btor2::{
    check_btor2, check_btor2_adaptive, parse, translate_to_chc, Btor2CheckResult, Btor2Error,
    Btor2Line, Btor2Node, Btor2Program, Btor2Sort,
};

// ---------------------------------------------------------------------------
// check_btor2_adaptive — preferred solve entry point
// ---------------------------------------------------------------------------

/// A program with no `bad` property short-circuits to an empty verdict vector
/// without invoking the solver at all.
#[test]
fn adaptive_no_bad_properties_returns_empty() {
    let src = "\
1 sort bitvec 4
2 state 1 s
";
    let program = parse(src).expect("should parse");
    assert!(program.bad_properties.is_empty());

    let results = check_btor2_adaptive(&program, Some(Duration::from_secs(5)))
        .expect("translation should succeed");
    assert!(
        results.is_empty(),
        "no bad properties must yield an empty verdict vector, got {} results",
        results.len()
    );
}

/// A 1-bit register pinned to 0 with `bad = (r == 1)` is unreachable: the
/// adaptive path must return `Unsat`. This also exercises the proof-backed-SAFE
/// contract (the discovered invariant is independently re-verified before being
/// reported), so an `Unknown` here would be a real regression of that contract.
#[test]
fn adaptive_trivially_safe_is_unsat() {
    // r := 0; next r := 0; bad = r (the 1-bit register value itself).
    let src = "\
1 sort bitvec 1
2 zero 1
3 state 1 r
4 init 1 3 2
5 next 1 3 3
6 bad 3
";
    let program = parse(src).expect("should parse");

    let results = check_btor2_adaptive(&program, Some(Duration::from_secs(30)))
        .expect("translation should succeed");
    assert_eq!(results.len(), 1);
    match &results[0] {
        Btor2CheckResult::Unsat => {}
        other => panic!("expected proof-backed Unsat (safe), got: {other:?}"),
    }
}

/// An 8-bit counter that starts at 0, increments by 1, with `bad = (count == 3)`
/// has a reachable bad state at depth 3. The adaptive path must return `Sat`
/// with a non-empty counterexample trace.
#[test]
fn adaptive_shallow_bug_is_sat_with_trace() {
    let src = "\
1 sort bitvec 8
2 sort bitvec 1
3 zero 1
4 state 1 count
5 init 1 4 3
6 one 1
7 add 1 4 6
8 next 1 4 7
9 constd 1 3
10 eq 2 4 9
11 bad 10
";
    let program = parse(src).expect("should parse");

    let results = check_btor2_adaptive(&program, Some(Duration::from_secs(30)))
        .expect("translation should succeed");
    assert_eq!(results.len(), 1);
    match &results[0] {
        Btor2CheckResult::Sat { trace, .. } => {
            assert!(
                !trace.is_empty(),
                "a reachable bad state must yield a non-empty counterexample trace"
            );
        }
        other => panic!("expected Sat (reachable bad state), got: {other:?}"),
    }
}

/// `check_btor2_adaptive` lowers through the same `translate_to_chc` path, so a
/// hand-built program with an over-wide sort must surface that error rather than
/// a verdict over a truncated model.
#[test]
fn adaptive_rejects_oversized_width() {
    let program = oversized_width_program();
    let result = check_btor2_adaptive(&program, Some(Duration::from_secs(5)));
    assert!(
        matches!(result, Err(Btor2Error::ParseError { .. })),
        "adaptive must decline an over-wide bitvector, got: {result:?}"
    );
}

// ---------------------------------------------------------------------------
// translate_to_chc — public CHC lowering
// ---------------------------------------------------------------------------

/// The lowering produces exactly the 3-clause skeleton (init fact, transition,
/// query) for a single-property program, and records one state var and one
/// property index.
#[test]
fn translate_single_property_clause_structure() {
    let src = "\
1 sort bitvec 1
2 zero 1
3 state 1 r
4 init 1 3 2
5 next 1 3 3
6 bad 3
";
    let program = parse(src).expect("should parse");
    let result = translate_to_chc(&program).expect("translation should succeed");

    assert_eq!(result.state_vars.len(), 1);
    assert_eq!(result.state_vars[0].name.as_deref(), Some("r"));
    // The current- and next-state vars must be distinct (priming actually applied).
    assert_ne!(
        result.state_vars[0].var, result.state_vars[0].next_var,
        "current and next state CHC variables must differ"
    );

    // init (fact) + transition + 1 query.
    assert_eq!(result.problem.clauses().len(), 3);
    assert_eq!(result.property_indices.len(), 1);
    assert!(
        result.problem.clauses()[0].is_fact(),
        "init clause is a fact"
    );
    assert!(
        !result.problem.clauses()[1].is_fact(),
        "transition clause is not a fact"
    );
    // The recorded property index points at a query clause.
    let q = result.property_indices[0];
    assert!(
        result.problem.clauses()[q].is_query(),
        "property index must point at a query clause"
    );
}

/// Two `bad` properties produce two query clauses (init + transition + 2
/// queries) and two distinct property indices, in property order.
#[test]
fn translate_multiple_properties_each_get_a_query() {
    let src = "\
1 sort bitvec 8
2 sort bitvec 1
3 zero 1
4 state 1 count
5 init 1 4 3
6 one 1
7 add 1 4 6
8 next 1 4 7
9 constd 1 3
10 eq 2 4 9
11 constd 1 7
12 eq 2 4 11
13 bad 10
14 bad 12
";
    let program = parse(src).expect("should parse");
    assert_eq!(program.bad_properties, vec![13, 14]);

    let result = translate_to_chc(&program).expect("translation should succeed");
    // init + transition + 2 queries.
    assert_eq!(result.problem.clauses().len(), 4);
    assert_eq!(result.property_indices.len(), 2);
    // Each recorded index must be a distinct query clause.
    assert_ne!(result.property_indices[0], result.property_indices[1]);
    for &idx in &result.property_indices {
        assert!(
            result.problem.clauses()[idx].is_query(),
            "property index {idx} must reference a query clause"
        );
    }
}

/// A bitvector sort wider than [`MAX_BV_WIDTH`] cannot be modeled by the
/// u128-backed encoding, so the lowering must decline (documented `# Errors`).
#[test]
fn translate_rejects_oversized_width() {
    let program = oversized_width_program();
    let result = translate_to_chc(&program);
    assert!(
        matches!(result, Err(Btor2Error::ParseError { .. })),
        "translate_to_chc must decline an over-wide bitvector, got: {result:?}"
    );
}

/// A `bad` property referencing a node id that is not defined must surface
/// `UndefinedNode` rather than panicking or silently dropping the property.
#[test]
fn translate_undefined_node_reference_errors() {
    // bad references node 99 which is never defined.
    let mut sorts = HashMap::new();
    sorts.insert(1, Btor2Sort::BitVec(1));
    let program = Btor2Program {
        lines: vec![
            Btor2Line {
                id: 1,
                sort_id: 0,
                node: Btor2Node::SortBitVec(1),
                args: vec![],
            },
            Btor2Line {
                id: 2,
                sort_id: 0,
                node: Btor2Node::Bad(99),
                args: vec![99],
            },
        ],
        sorts,
        num_inputs: 0,
        num_states: 0,
        bad_properties: vec![2],
        constraints: vec![],
        fairness: vec![],
        justice: vec![],
    };

    let result = translate_to_chc(&program);
    assert!(
        matches!(result, Err(Btor2Error::UndefinedNode { node_id: 99, .. })),
        "expected UndefinedNode(99), got: {result:?}"
    );
}

// ---------------------------------------------------------------------------
// check_btor2 — simple translator empty-property contract
// ---------------------------------------------------------------------------

/// `check_btor2` on a program with no `bad` property returns an empty vector
/// (no solver invocation), matching its documented contract.
#[test]
fn check_btor2_no_bad_properties_returns_empty() {
    let src = "\
1 sort bitvec 4
2 input 1 x
3 state 1 s
";
    let program = parse(src).expect("should parse");
    let results = check_btor2(&program).expect("translation should succeed");
    assert!(results.is_empty());
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A hand-built program whose sole sort exceeds [`MAX_BV_WIDTH`]. The text
/// parser rejects such a width up front, so the program is constructed directly
/// to reach the lowering's own width guard.
fn oversized_width_program() -> Btor2Program {
    let wide = MAX_BV_WIDTH + 1;
    let mut sorts = HashMap::new();
    sorts.insert(1, Btor2Sort::BitVec(wide));
    Btor2Program {
        lines: vec![
            Btor2Line {
                id: 1,
                sort_id: 0,
                node: Btor2Node::SortBitVec(wide),
                args: vec![],
            },
            Btor2Line {
                id: 2,
                sort_id: 1,
                node: Btor2Node::Zero,
                args: vec![],
            },
            Btor2Line {
                id: 3,
                sort_id: 1,
                node: Btor2Node::State(1, Some("s".to_string())),
                args: vec![],
            },
            Btor2Line {
                id: 4,
                sort_id: 1,
                node: Btor2Node::Init(1, 3, 2),
                args: vec![3, 2],
            },
            Btor2Line {
                id: 5,
                sort_id: 1,
                node: Btor2Node::Next(1, 3, 3),
                args: vec![3, 3],
            },
            Btor2Line {
                id: 6,
                sort_id: 0,
                node: Btor2Node::Bad(3),
                args: vec![3],
            },
        ],
        sorts,
        num_inputs: 0,
        num_states: 1,
        bad_properties: vec![6],
        constraints: vec![],
        fairness: vec![],
        justice: vec![],
    }
}
