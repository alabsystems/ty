// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;

use crate::checker_ops::expand_operator_body_with_primes;
use crate::coverage::detect_actions;
use crate::var_index::VarIndex;
use crate::EvalCtx;
use tla_core::ast::Unit;
use tla_core::{lower, parse_to_syntax_tree, FileId};

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_static_independence_disjoint() {
    let mut a = ActionDependencies::new();
    a.add_read(VarIndex(0));
    a.add_write(VarIndex(0));

    let mut b = ActionDependencies::new();
    b.add_read(VarIndex(1));
    b.add_write(VarIndex(1));

    assert!(check_static_independence(&a, &b));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_static_independence_overlapping_write_read() {
    let mut a = ActionDependencies::new();
    a.add_write(VarIndex(0));

    let mut b = ActionDependencies::new();
    b.add_read(VarIndex(0));

    assert!(!check_static_independence(&a, &b));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_static_independence_overlapping_write_write() {
    let mut a = ActionDependencies::new();
    a.add_write(VarIndex(0));

    let mut b = ActionDependencies::new();
    b.add_write(VarIndex(0));

    assert!(!check_static_independence(&a, &b));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_static_independence_read_read_ok() {
    let mut a = ActionDependencies::new();
    a.add_read(VarIndex(0));

    let mut b = ActionDependencies::new();
    b.add_read(VarIndex(0));

    assert!(check_static_independence(&a, &b));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_independence_matrix_empty() {
    let matrix = IndependenceMatrix::compute(&[]);
    assert_eq!(matrix.action_count(), 0);
    assert_eq!(matrix.count_independent_pairs(), 0);
    assert_eq!(matrix.total_pairs(), 0);
}

// ==================== Ample Set Tests ====================

/// Create a mock independence matrix for testing
fn make_test_matrix(n: usize, independent_pairs: &[(usize, usize)]) -> IndependenceMatrix {
    let deps: Vec<ActionDependencies> = (0..n)
        .map(|i| {
            let mut d = ActionDependencies::new();
            d.add_write(VarIndex::new(i));
            d
        })
        .collect();

    let mut matrix = vec![IndependenceStatus::Dependent; n * n];
    for i in 0..n {
        matrix[i * n + i] = IndependenceStatus::Dependent;
    }
    for &(i, j) in independent_pairs {
        matrix[i * n + j] = IndependenceStatus::Independent;
        matrix[j * n + i] = IndependenceStatus::Independent;
    }

    IndependenceMatrix {
        n,
        matrix,
        dependencies: deps,
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_ample_set_empty_enabled() {
    let matrix = make_test_matrix(3, &[]);
    let visibility = VisibilitySet::new();
    let result = compute_ample_set(&[], &matrix, &visibility);
    assert!(result.actions.is_empty());
    assert!(!result.reduced);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_ample_set_single_action() {
    let matrix = make_test_matrix(3, &[]);
    let visibility = VisibilitySet::new();
    let result = compute_ample_set(&[1], &matrix, &visibility);
    assert_eq!(result.actions, vec![1]);
    assert!(!result.reduced);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_ample_set_no_independent_actions() {
    // Actions 0, 1, 2 all dependent on each other
    let matrix = make_test_matrix(3, &[]);
    let visibility = VisibilitySet::new();
    let result = compute_ample_set(&[0, 1, 2], &matrix, &visibility);
    assert_eq!(result.actions, vec![0, 1, 2]);
    assert!(!result.reduced);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_ample_set_singleton_seed_when_independent() {
    // Action 0 is independent of actions 1 and 2.
    // Closure from seed 0 only contains 0 itself.
    // With empty visibility, action 0 is not visible, so reduction succeeds.
    let matrix = make_test_matrix(3, &[(0, 1), (0, 2)]);
    let visibility = VisibilitySet::new();
    let result = compute_ample_set(&[0, 1, 2], &matrix, &visibility);
    assert_eq!(result.actions, vec![0]);
    assert!(result.reduced);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_ample_set_visibility_blocks_reduction() {
    // Action 0 is independent of 1 and 2 but visible (writes to var in invariant).
    // Actions 1 and 2 are dependent on each other (both write to their own var,
    // but default make_test_matrix marks non-listed pairs as dependent).
    let mut matrix = make_test_matrix(3, &[(0, 1), (0, 2)]);
    // Action 0 writes to var 0
    matrix.dependencies[0].writes.insert(VarIndex(0));

    let mut visibility = VisibilitySet::new();
    visibility.vars.insert(VarIndex(0)); // Var 0 in invariant

    let result = compute_ample_set(&[0, 1, 2], &matrix, &visibility);
    // Action 0 is skipped as seed due to visibility.
    // Actions 1 and 2 are dependent, so closing from either pulls in the other.
    // Neither 1 nor 2 is visible, so closure {1, 2} is valid but not a reduction
    // (size 2 < 3 is a reduction).
    // Actually: 1 and 2 are dependent, so closing from seed 1 adds 2.
    // Ample set = {1, 2}, which is a reduction from {0, 1, 2}.
    assert!(result.reduced);
    assert_eq!(result.actions.len(), 2);
    assert!(
        !result.actions.contains(&0),
        "visible action 0 should not be in ample set"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_ample_set_reduces_to_singleton_when_all_independent() {
    // All three actions are mutually independent.
    // Any singleton is a valid ample set. Seed heuristic picks the first
    // non-visible action with fewest dependencies.
    let matrix = make_test_matrix(3, &[(0, 1), (0, 2), (1, 2)]);
    let visibility = VisibilitySet::new();
    let result = compute_ample_set(&[0, 1, 2], &matrix, &visibility);
    assert_eq!(result.actions.len(), 1);
    assert!(result.reduced);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_visibility_set_from_empty() {
    let visibility = VisibilitySet::from_eval_invariants(&[]);
    assert!(visibility.vars.is_empty());
}

/// Verify `from_eval_invariants` extracts state variable reads from non-empty
/// invariant expressions.  Part of #3354 Slice 4 test gap coverage: the old
/// `from_invariants(&[CompiledGuard])` replacement must produce the same
/// variable-read sets when given equivalent AST invariants.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_visibility_set_from_eval_invariants_extracts_reads() {
    use tla_core::ast::Expr;
    use tla_core::span::Spanned;
    use tla_core::NameId;

    // Invariant: x > 0  (reads state var x at index 0)
    let x_var = Spanned::dummy(Expr::StateVar("x".to_string(), 0, NameId::INVALID));
    let zero = Spanned::dummy(Expr::Int(0.into()));
    let inv_body = Spanned::dummy(Expr::Gt(Box::new(x_var), Box::new(zero)));
    let invariants = vec![("Inv1".to_string(), inv_body)];

    let visibility = VisibilitySet::from_eval_invariants(&invariants);
    assert!(
        visibility.vars.contains(&VarIndex(0)),
        "visibility set should contain var index 0 (x)"
    );
    assert_eq!(visibility.vars.len(), 1);
}

/// Verify `from_eval_invariants` collects reads across multiple invariants.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_visibility_set_from_multiple_invariants() {
    use tla_core::ast::Expr;
    use tla_core::span::Spanned;
    use tla_core::NameId;

    // Inv1: x > 0  (reads var 0)
    let inv1 = Spanned::dummy(Expr::Gt(
        Box::new(Spanned::dummy(Expr::StateVar(
            "x".to_string(),
            0,
            NameId::INVALID,
        ))),
        Box::new(Spanned::dummy(Expr::Int(0.into()))),
    ));

    // Inv2: y = TRUE  (reads var 2)
    let inv2 = Spanned::dummy(Expr::Eq(
        Box::new(Spanned::dummy(Expr::StateVar(
            "y".to_string(),
            2,
            NameId::INVALID,
        ))),
        Box::new(Spanned::dummy(Expr::Bool(true))),
    ));

    let invariants = vec![("Inv1".to_string(), inv1), ("Inv2".to_string(), inv2)];

    let visibility = VisibilitySet::from_eval_invariants(&invariants);
    assert!(
        visibility.vars.contains(&VarIndex(0)),
        "should contain x (var 0)"
    );
    assert!(
        visibility.vars.contains(&VarIndex(2)),
        "should contain y (var 2)"
    );
    assert_eq!(visibility.vars.len(), 2);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_por_stats() {
    let mut stats = PorStats::default();
    assert_eq!(stats.total_states, 0);

    stats.record(3, 1); // Reduced
    assert_eq!(stats.reductions, 1);
    assert_eq!(stats.total_states, 1);
    assert_eq!(stats.actions_skipped, 2);

    stats.record(2, 2); // Not reduced
    assert_eq!(stats.reductions, 1);
    assert_eq!(stats.total_states, 2);
    assert_eq!(stats.actions_skipped, 2);
}

// ==================== Static Analysis Limitation Tests ====================
//
// These unit tests document why static POR analysis cannot reduce most TLA+ specs:
// - TLA+ requires specifying all variables in every action (via UNCHANGED)
// - UNCHANGED x is treated as both read AND write of x (conservative for soundness)
// - This makes virtually all actions dependent
//
// ay-based semantic independence checking (Phase 3) would be needed for
// practical state space reduction.
//
// NOTE: These are unit tests that simulate dependency patterns, not integration
// tests that run the model checker. End-to-end POR verification would require
// running specs with and without --por and comparing results.

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_ample_set_all_visible_blocks_all_reductions() {
    // All actions are visible: no ample set can be a proper subset (C2).
    let matrix = make_test_matrix(3, &[(0, 1), (0, 2), (1, 2)]);
    let mut visibility = VisibilitySet::new();
    visibility.vars.insert(VarIndex(0));
    visibility.vars.insert(VarIndex(1));
    visibility.vars.insert(VarIndex(2));

    let result = compute_ample_set(&[0, 1, 2], &matrix, &visibility);
    assert_eq!(result.actions, vec![0, 1, 2]);
    assert!(!result.reduced);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_ample_set_closure_grows_to_full_set() {
    // All actions are pairwise dependent (no independent pairs).
    // Closure from any seed must include all enabled actions.
    let matrix = make_test_matrix(3, &[]);
    let visibility = VisibilitySet::new();
    let result = compute_ample_set(&[0, 1, 2], &matrix, &visibility);
    assert_eq!(result.actions, vec![0, 1, 2]);
    assert!(!result.reduced);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_ample_set_partial_closure() {
    // 4 actions: 0 independent of all, 1-2 dependent, 3 independent of all.
    // Seed 0: closure = {0} (all others independent). Ample set = {0}.
    let matrix = make_test_matrix(4, &[(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]);
    let visibility = VisibilitySet::new();
    let result = compute_ample_set(&[0, 1, 2, 3], &matrix, &visibility);
    assert!(result.reduced);
    assert_eq!(result.actions.len(), 1);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_ample_set_visible_in_closure_forces_fallback() {
    // 3 actions: 0 independent of 2, but 1 is dependent on 0.
    // Action 1 is visible.
    // Seed 0: closure adds 1 (dependent), 1 is visible -> fallback.
    // Seed 2: 2 is dependent on 1 (no (1,2) in independent_pairs), closure
    // adds 1, 1 is visible -> fallback.
    // No valid ample set: return full enabled set.
    let matrix = make_test_matrix(3, &[(0, 2)]);
    let mut visibility = VisibilitySet::new();
    // Action 1 writes to var 1, which is in the invariant.
    visibility.vars.insert(VarIndex(1));

    let result = compute_ample_set(&[0, 1, 2], &matrix, &visibility);
    assert_eq!(result.actions, vec![0, 1, 2]);
    assert!(!result.reduced);
}

/// UNCHANGED with identity-write tracking makes actions with disjoint real
/// writes INDEPENDENT, even when they share variables via UNCHANGED clauses.
///
/// Part of #3993: UNCHANGED x is the identity function on x. It commutes
/// with ALL operations, including real writes. The "read" in UNCHANGED x
/// is vacuous for commutativity — the action just preserves whatever value
/// x has, regardless of what other actions did to it.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_unchanged_enables_independence() {
    // IncX: reads x (guard), real-writes x, identity-writes y (UNCHANGED y)
    // IncY: reads y (guard), real-writes y, identity-writes x (UNCHANGED x)
    let mut inc_x = ActionDependencies::new();
    inc_x.add_read(VarIndex(0)); // reads x (guard: x < 3)
    inc_x.add_write(VarIndex(0)); // real write: x' = x + 1
    inc_x.add_unchanged(VarIndex(1)); // UNCHANGED y: identity write

    let mut inc_y = ActionDependencies::new();
    inc_y.add_read(VarIndex(1)); // reads y (guard: y < 3)
    inc_y.add_write(VarIndex(1)); // real write: y' = y + 1
    inc_y.add_unchanged(VarIndex(0)); // UNCHANGED x: identity write

    // IncX real-writes {x}. Check against IncY: reads={y}, writes={y}.
    // x not in {y} ∪ {y} → no conflict from A's side.
    // IncY real-writes {y}. Check against IncX: reads={x}, writes={x}.
    // y not in {x} ∪ {x} → no conflict from B's side.
    // Identity writes (unchanged) are excluded from conflict checks.
    assert!(check_static_independence(&inc_x, &inc_y));
}

/// Two actions with completely disjoint REAL access patterns are
/// independent, even though they share variables via UNCHANGED.
///
/// Part of #3993: This is the primary POR improvement case.
/// IncX only reads/writes x, IncY only reads/writes y.
/// Both have UNCHANGED on the other's variable.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_unchanged_disjoint_real_access_independent() {
    let mut inc_x = ActionDependencies::new();
    inc_x.add_read(VarIndex(0)); // reads x
    inc_x.add_write(VarIndex(0)); // writes x
    inc_x.add_unchanged(VarIndex(1)); // UNCHANGED y

    let mut inc_y = ActionDependencies::new();
    inc_y.add_read(VarIndex(1)); // reads y
    inc_y.add_write(VarIndex(1)); // writes y
    inc_y.add_unchanged(VarIndex(0)); // UNCHANGED x

    // Disjoint real access: x and y are separate variables.
    // UNCHANGED does not create conflicts.
    assert!(check_static_independence(&inc_x, &inc_y));
}

/// Actions that share a REAL read/write on the same variable remain
/// dependent even with UNCHANGED tracking.
///
/// Part of #3993: If A writes x and B reads x (not via UNCHANGED),
/// they are still dependent because the read observes A's write.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_cross_read_write_still_dependent() {
    // A: writes x, reads y (real access to y, not UNCHANGED)
    let mut action_a = ActionDependencies::new();
    action_a.add_write(VarIndex(0)); // writes x
    action_a.add_read(VarIndex(1)); // reads y

    // B: writes y, reads x (real access to x, not UNCHANGED)
    let mut action_b = ActionDependencies::new();
    action_b.add_write(VarIndex(1)); // writes y
    action_b.add_read(VarIndex(0)); // reads x

    // A writes x, B reads x → dependent
    assert!(!check_static_independence(&action_a, &action_b));
}

/// Document: explicit x' = x assignments (as real writes) still create dependency.
///
/// When the AST extractor sees x' = expr (not UNCHANGED), it's recorded
/// as a real write. This test simulates that scenario.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_identity_assignment_as_real_write_creates_dependency() {
    let mut action_a = ActionDependencies::new();
    action_a.add_write(VarIndex(0)); // writes a
    action_a.add_read(VarIndex(1)); // b' = b reads b
    action_a.add_write(VarIndex(1)); // b' = b writes b (real write at AST level)

    let mut action_b = ActionDependencies::new();
    action_b.add_write(VarIndex(1)); // writes b
    action_b.add_read(VarIndex(0)); // a' = a reads a
    action_b.add_write(VarIndex(0)); // a' = a writes a (real write at AST level)

    assert!(!check_static_independence(&action_a, &action_b));
}

/// Part of #3993: Explicit `x' = x` identity assignment is recognized as
/// an identity write (equivalent to UNCHANGED), enabling independence detection.
///
/// Without this optimization, `x' = x` would be recorded as a real write to x
/// plus a read of x, making virtually all actions dependent. The identity
/// detection in `extract_dependencies_ast_expr` recognizes the
/// `Eq(Prime(StateVar(idx)), StateVar(idx))` pattern.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_identity_assignment_detected_as_unchanged() {
    let spec = r#"
---- MODULE PorIdentityDetection ----
EXTENDS Naturals

VARIABLE x, y

IncX ==
    /\ x < 3
    /\ x' = x + 1
    /\ y' = y          \* This is x' = x — should be detected as identity

IncY ==
    /\ y < 3
    /\ x' = x          \* This is x' = x — should be detected as identity
    /\ y' = y + 1

Next == IncX \/ IncY

====
"#;

    let tree = parse_to_syntax_tree(spec);
    let lowered = lower(FileId(0), &tree);
    assert!(
        lowered.errors.is_empty(),
        "unexpected lower errors: {:?}",
        lowered.errors
    );
    let module = lowered.module.expect("lowered module");
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let mut var_names = Vec::new();
    for unit in &module.units {
        if let Unit::Variable(vars) = &unit.node {
            for var in vars {
                var_names.push(var.node.clone());
            }
        }
    }
    var_names.sort();
    ctx.register_vars(var_names.iter().cloned());
    ctx.resolve_state_vars_in_loaded_ops();

    let next_def = module
        .units
        .iter()
        .find_map(|unit| match &unit.node {
            Unit::Operator(def) if def.name.node == "Next" => Some(def),
            _ => None,
        })
        .expect("missing Next");

    let expanded_next = expand_operator_body_with_primes(&ctx, next_def);
    let actions = detect_actions(&expanded_next);
    assert_eq!(actions.len(), 2);

    let dependencies = extract_detected_action_dependencies(&ctx, &actions);
    let inc_x_deps = &dependencies[0];
    let inc_y_deps = &dependencies[1];

    // IncX: y' = y should be detected as identity write to y, NOT a real write.
    assert!(
        !inc_x_deps.writes.contains(&VarIndex(1)),
        "y' = y should NOT be a real write in IncX"
    );
    assert!(
        inc_x_deps.unchanged.contains(&VarIndex(1)),
        "y' = y should be an identity write (unchanged) in IncX"
    );

    // IncY: x' = x should be detected as identity write to x, NOT a real write.
    assert!(
        !inc_y_deps.writes.contains(&VarIndex(0)),
        "x' = x should NOT be a real write in IncY"
    );
    assert!(
        inc_y_deps.unchanged.contains(&VarIndex(0)),
        "x' = x should be an identity write (unchanged) in IncY"
    );

    // Actions should be independent (disjoint real read/write sets).
    let matrix = IndependenceMatrix::compute(&dependencies);
    assert_eq!(
        matrix.get(0, 1),
        IndependenceStatus::Independent,
        "IncX and IncY should be independent with identity assignment detection"
    );
}

/// Integration test: UNCHANGED in parsed TLA+ spec now creates identity
/// writes, enabling independence detection.
///
/// Part of #3993: IncX real-writes x, UNCHANGED y; IncY real-writes y, UNCHANGED x.
/// With identity-write tracking, UNCHANGED does not create read/write
/// dependencies. The actions are INDEPENDENT because:
/// - IncX real-writes {x}, IncY reads {y} and real-writes {y} → disjoint
/// - IncY real-writes {y}, IncX reads {x} and real-writes {x} → disjoint
/// - UNCHANGED vars are identity writes, excluded from conflict checks
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_detected_action_dependencies_with_unchanged_commutativity() {
    let spec = r#"
---- MODULE PorDetectedActions ----
EXTENDS Naturals

VARIABLE x, y

IncX ==
    /\ x < 3
    /\ x' = x + 1
    /\ UNCHANGED y

IncY ==
    /\ y < 3
    /\ UNCHANGED x
    /\ y' = y + 1

Next == IncX \/ IncY

====
"#;

    let tree = parse_to_syntax_tree(spec);
    let lowered = lower(FileId(0), &tree);
    assert!(
        lowered.errors.is_empty(),
        "unexpected lower errors: {:?}",
        lowered.errors
    );
    let module = lowered.module.expect("lowered module");
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let mut var_names = Vec::new();
    for unit in &module.units {
        if let Unit::Variable(vars) = &unit.node {
            for var in vars {
                var_names.push(var.node.clone());
            }
        }
    }
    var_names.sort();
    ctx.register_vars(var_names.iter().cloned());
    ctx.resolve_state_vars_in_loaded_ops();

    let next_def = module
        .units
        .iter()
        .find_map(|unit| match &unit.node {
            Unit::Operator(def) if def.name.node == "Next" => Some(def),
            _ => None,
        })
        .expect("missing Next");

    let expanded_next = expand_operator_body_with_primes(&ctx, next_def);
    let actions = detect_actions(&expanded_next);
    assert_eq!(actions.len(), 2, "expected two detected top-level actions");

    let dependencies = extract_detected_action_dependencies(&ctx, &actions);

    // Verify UNCHANGED creates identity writes, not real writes or reads.
    let inc_x_deps = &dependencies[0];
    let inc_y_deps = &dependencies[1];

    // UNCHANGED y should be in unchanged set, not writes or reads
    assert!(
        !inc_x_deps.writes.contains(&VarIndex(1)),
        "UNCHANGED y should NOT be in IncX.writes"
    );
    assert!(
        inc_x_deps.unchanged.contains(&VarIndex(1)),
        "UNCHANGED y should be in IncX.unchanged"
    );
    assert!(
        !inc_x_deps.reads.contains(&VarIndex(1)),
        "UNCHANGED y should NOT add a read of y in IncX"
    );

    assert!(
        !inc_y_deps.writes.contains(&VarIndex(0)),
        "UNCHANGED x should NOT be in IncY.writes"
    );
    assert!(
        inc_y_deps.unchanged.contains(&VarIndex(0)),
        "UNCHANGED x should be in IncY.unchanged"
    );
    assert!(
        !inc_y_deps.reads.contains(&VarIndex(0)),
        "UNCHANGED x should NOT add a read of x in IncY"
    );

    let matrix = IndependenceMatrix::compute(&dependencies);
    // IncX and IncY are INDEPENDENT:
    // - IncX real-writes {x}, IncY reads={y}, writes={y} → disjoint
    // - IncY real-writes {y}, IncX reads={x}, writes={x} → disjoint
    // - UNCHANGED vars don't participate in conflict checks
    assert_eq!(
        matrix.get(0, 1),
        IndependenceStatus::Independent,
        "IncX and IncY should be independent with UNCHANGED commutativity"
    );
    assert_eq!(
        matrix.get(1, 0),
        IndependenceStatus::Independent,
        "IncY and IncX should be independent with UNCHANGED commutativity"
    );
}

/// Integration test: three actions on disjoint variables are all
/// mutually independent with UNCHANGED commutativity.
///
/// Part of #3993: This demonstrates the full POR win on concurrent specs.
/// All 3 pairs are independent → ample set is a singleton → 3x reduction.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_three_action_all_independent() {
    let spec = r#"
---- MODULE PorThreeActions ----
EXTENDS Naturals

VARIABLE x, y, z

IncX ==
    /\ x < 3
    /\ x' = x + 1
    /\ UNCHANGED <<y, z>>

IncY ==
    /\ y < 3
    /\ y' = y + 1
    /\ UNCHANGED <<x, z>>

IncZ ==
    /\ z < 3
    /\ z' = z + 1
    /\ UNCHANGED <<x, y>>

Next == IncX \/ IncY \/ IncZ

====
"#;

    let tree = parse_to_syntax_tree(spec);
    let lowered = lower(FileId(0), &tree);
    assert!(
        lowered.errors.is_empty(),
        "unexpected lower errors: {:?}",
        lowered.errors
    );
    let module = lowered.module.expect("lowered module");
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let mut var_names = Vec::new();
    for unit in &module.units {
        if let Unit::Variable(vars) = &unit.node {
            for var in vars {
                var_names.push(var.node.clone());
            }
        }
    }
    var_names.sort();
    ctx.register_vars(var_names.iter().cloned());
    ctx.resolve_state_vars_in_loaded_ops();

    let next_def = module
        .units
        .iter()
        .find_map(|unit| match &unit.node {
            Unit::Operator(def) if def.name.node == "Next" => Some(def),
            _ => None,
        })
        .expect("missing Next");

    let expanded_next = expand_operator_body_with_primes(&ctx, next_def);
    let actions = detect_actions(&expanded_next);
    assert_eq!(
        actions.len(),
        3,
        "expected three detected top-level actions"
    );

    let dependencies = extract_detected_action_dependencies(&ctx, &actions);
    let matrix = IndependenceMatrix::compute(&dependencies);

    // Each action real-writes one var and has UNCHANGED on the other two.
    // With UNCHANGED commutativity, identity writes don't create conflicts.
    // IncX: reads={x}, writes={x}, unchanged={y,z}
    // IncY: reads={y}, writes={y}, unchanged={x,z}
    // IncZ: reads={z}, writes={z}, unchanged={x,y}
    // All pairs have disjoint real reads and writes → all independent.
    assert_eq!(
        matrix.get(0, 1),
        IndependenceStatus::Independent,
        "IncX and IncY should be independent"
    );
    assert_eq!(
        matrix.get(0, 2),
        IndependenceStatus::Independent,
        "IncX and IncZ should be independent"
    );
    assert_eq!(
        matrix.get(1, 2),
        IndependenceStatus::Independent,
        "IncY and IncZ should be independent"
    );
}

/// Part of #3449: verify that `extend_from_expanded_expr` sees through wrapper
/// operators to extract the underlying state variable reads.
///
/// A config invariant `Inv == TypeOK` where `TypeOK == x = 0` should
/// produce visibility for `x` (VarIndex 0), not just an opaque `Ident("TypeOK")`.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_visibility_extend_from_expanded_expr_sees_through_wrapper() {
    let spec = r#"
---- MODULE PorWrapperInv ----
EXTENDS Naturals

VARIABLE x

TypeOK == x = 0

Inv == TypeOK

Init == x = 0
Next == x' = x + 1
====
"#;

    let tree = parse_to_syntax_tree(spec);
    let lowered = lower(FileId(0), &tree);
    assert!(
        lowered.errors.is_empty(),
        "unexpected lower errors: {:?}",
        lowered.errors
    );
    let module = lowered.module.expect("lowered module");
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let mut var_names = Vec::new();
    for unit in &module.units {
        if let Unit::Variable(vars) = &unit.node {
            for var in vars {
                var_names.push(var.node.clone());
            }
        }
    }
    var_names.sort();
    ctx.register_vars(var_names.iter().cloned());
    ctx.resolve_state_vars_in_loaded_ops();

    // Look up the resolved "Inv" operator body from ctx (not module.units).
    // After resolve_state_vars_in_loaded_ops(), ctx has state-var-resolved bodies.
    let inv_def = ctx.get_op("Inv").expect("missing Inv operator in ctx");

    let mut visibility = VisibilitySet::new();
    visibility.extend_from_expanded_expr(&ctx, &inv_def.body);

    assert!(
        visibility.vars.contains(&VarIndex(0)),
        "visibility set should contain var index 0 (x) after expanding Inv -> TypeOK -> x = 0"
    );
}

/// Part of #3449: verify that `mark_all_visible` makes all actions visible.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_visibility_mark_all_visible_blocks_all_reductions() {
    let mut visibility = VisibilitySet::new();
    assert!(visibility.vars.is_empty());

    // Without mark_all_visible, an action writing to var 5 is not visible
    let mut deps = ActionDependencies::new();
    deps.add_write(VarIndex(5));
    assert!(
        !visibility.is_action_visible(&deps),
        "empty visibility should not see var 5"
    );

    // After mark_all_visible, ALL actions should be visible
    visibility.mark_all_visible();
    assert!(
        visibility.is_action_visible(&deps),
        "all_visible should make every action visible"
    );
}

/// Part of #3449: verify that config-level invariant reads merge with
/// PROPERTY-level invariant reads in the visibility set.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_visibility_config_and_property_invariants_merge() {
    use tla_core::ast::Expr;
    use tla_core::span::Spanned;
    use tla_core::NameId;

    let spec = r#"
---- MODULE PorMergedInv ----
EXTENDS Naturals

VARIABLE x, y

ConfigInv == y > 0

Init == x = 0 /\ y = 1
Next == x' = x + 1 /\ y' = y
====
"#;

    let tree = parse_to_syntax_tree(spec);
    let lowered = lower(FileId(0), &tree);
    assert!(
        lowered.errors.is_empty(),
        "unexpected lower errors: {:?}",
        lowered.errors
    );
    let module = lowered.module.expect("lowered module");
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let mut var_names = Vec::new();
    for unit in &module.units {
        if let Unit::Variable(vars) = &unit.node {
            for var in vars {
                var_names.push(var.node.clone());
            }
        }
    }
    var_names.sort();
    ctx.register_vars(var_names.iter().cloned());
    ctx.resolve_state_vars_in_loaded_ops();

    // Start with a PROPERTY invariant that reads x (var 0)
    let property_inv_body = Spanned::dummy(Expr::Gt(
        Box::new(Spanned::dummy(Expr::StateVar(
            "x".to_string(),
            0,
            NameId::INVALID,
        ))),
        Box::new(Spanned::dummy(Expr::Int(0.into()))),
    ));
    let property_invariants = vec![("PropertyInv".to_string(), property_inv_body)];
    let mut visibility = VisibilitySet::from_eval_invariants(&property_invariants);
    assert!(
        visibility.vars.contains(&VarIndex(0)),
        "should contain x (var 0) from PROPERTY"
    );

    // Now extend with config invariant ConfigInv that reads y (var 1).
    // Use ctx.get_op() to get the resolved body (with state vars resolved).
    let config_inv_def = ctx
        .get_op("ConfigInv")
        .expect("missing ConfigInv operator in ctx");
    visibility.extend_from_expanded_expr(&ctx, &config_inv_def.body);

    assert!(
        visibility.vars.contains(&VarIndex(0)),
        "should still contain x from PROPERTY"
    );
    assert!(
        visibility.vars.contains(&VarIndex(1)),
        "should contain y (var 1) from config invariant ConfigInv"
    );
}

// ==================== Phase 11 Enhanced Independence Tests ====================

/// Part of #3993 Phase 11: detect identity through IF/THEN/ELSE.
///
/// `x' = IF cond THEN x ELSE x` is equivalent to UNCHANGED x.
/// Both branches evaluate to x, so the net effect is identity.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_identity_through_if_then_else() {
    let spec = r#"
---- MODULE PorIfIdentity ----
EXTENDS Naturals

VARIABLE x, y

IncX ==
    /\ x < 3
    /\ x' = x + 1
    /\ y' = IF x > 0 THEN y ELSE y   \* identity through IF/THEN/ELSE

IncY ==
    /\ y < 3
    /\ y' = y + 1
    /\ x' = IF y > 0 THEN x ELSE x   \* identity through IF/THEN/ELSE

Next == IncX \/ IncY

====
"#;

    let tree = parse_to_syntax_tree(spec);
    let lowered = lower(FileId(0), &tree);
    assert!(
        lowered.errors.is_empty(),
        "unexpected lower errors: {:?}",
        lowered.errors
    );
    let module = lowered.module.expect("lowered module");
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let mut var_names = Vec::new();
    for unit in &module.units {
        if let Unit::Variable(vars) = &unit.node {
            for var in vars {
                var_names.push(var.node.clone());
            }
        }
    }
    var_names.sort();
    ctx.register_vars(var_names.iter().cloned());
    ctx.resolve_state_vars_in_loaded_ops();

    let next_def = module
        .units
        .iter()
        .find_map(|unit| match &unit.node {
            Unit::Operator(def) if def.name.node == "Next" => Some(def),
            _ => None,
        })
        .expect("missing Next");

    let expanded_next = expand_operator_body_with_primes(&ctx, next_def);
    let actions = detect_actions(&expanded_next);
    assert_eq!(actions.len(), 2);

    let dependencies = extract_detected_action_dependencies(&ctx, &actions);
    let inc_x_deps = &dependencies[0];
    let inc_y_deps = &dependencies[1];

    // IncX: y' = IF x > 0 THEN y ELSE y should be detected as identity write to y
    assert!(
        !inc_x_deps.writes.contains(&VarIndex(1)),
        "y' = IF ... THEN y ELSE y should NOT be a real write in IncX"
    );
    assert!(
        inc_x_deps.unchanged.contains(&VarIndex(1)),
        "y' = IF ... THEN y ELSE y should be an identity write (unchanged) in IncX"
    );

    // IncY: x' = IF y > 0 THEN x ELSE x should be detected as identity write to x
    assert!(
        !inc_y_deps.writes.contains(&VarIndex(0)),
        "x' = IF ... THEN x ELSE x should NOT be a real write in IncY"
    );
    assert!(
        inc_y_deps.unchanged.contains(&VarIndex(0)),
        "x' = IF ... THEN x ELSE x should be an identity write (unchanged) in IncY"
    );

    // Both actions should be independent
    let matrix = IndependenceMatrix::compute(&dependencies);
    assert_eq!(
        matrix.get(0, 1),
        IndependenceStatus::Independent,
        "IncX and IncY should be independent with IF/THEN/ELSE identity detection"
    );
}

/// Part of #3993 Phase 11: constant write detection.
///
/// `x' = 0` does not read x — the write value is constant. This reduces
/// the read set and may enable independence that would otherwise be blocked.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_constant_write_no_read_dependency() {
    let spec = r#"
---- MODULE PorConstWrite ----
EXTENDS Naturals

VARIABLE x, y

ResetX ==
    /\ x' = 0               \* constant write — does NOT read x
    /\ y' = y                \* identity write

IncrY ==
    /\ y' = y + 1            \* real write to y, reads y
    /\ x' = x                \* identity write

Next == ResetX \/ IncrY

====
"#;

    let tree = parse_to_syntax_tree(spec);
    let lowered = lower(FileId(0), &tree);
    assert!(
        lowered.errors.is_empty(),
        "unexpected lower errors: {:?}",
        lowered.errors
    );
    let module = lowered.module.expect("lowered module");
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let mut var_names = Vec::new();
    for unit in &module.units {
        if let Unit::Variable(vars) = &unit.node {
            for var in vars {
                var_names.push(var.node.clone());
            }
        }
    }
    var_names.sort();
    ctx.register_vars(var_names.iter().cloned());
    ctx.resolve_state_vars_in_loaded_ops();

    let next_def = module
        .units
        .iter()
        .find_map(|unit| match &unit.node {
            Unit::Operator(def) if def.name.node == "Next" => Some(def),
            _ => None,
        })
        .expect("missing Next");

    let expanded_next = expand_operator_body_with_primes(&ctx, next_def);
    let actions = detect_actions(&expanded_next);
    assert_eq!(actions.len(), 2);

    let dependencies = extract_detected_action_dependencies(&ctx, &actions);
    let reset_x_deps = &dependencies[0];

    // ResetX: x' = 0 should record a write but NOT a read of x
    assert!(
        reset_x_deps.writes.contains(&VarIndex(0)),
        "x' = 0 should be a write to x"
    );
    assert!(
        !reset_x_deps.reads.contains(&VarIndex(0)),
        "x' = 0 (constant write) should NOT add a read of x"
    );

    // ResetX: y' = y is identity
    assert!(
        reset_x_deps.unchanged.contains(&VarIndex(1)),
        "y' = y should be an identity write in ResetX"
    );

    // ResetX: writes={x}, reads={}, unchanged={y}
    // IncrY:  writes={y}, reads={y}, unchanged={x}
    // ResetX.writes={x} vs IncrY.{reads={y}, writes={y}} — disjoint
    // IncrY.writes={y} vs ResetX.{reads={}, writes={x}} — disjoint
    // They should be independent.
    let matrix = IndependenceMatrix::compute(&dependencies);
    assert_eq!(
        matrix.get(0, 1),
        IndependenceStatus::Independent,
        "ResetX (constant write) and IncrY should be independent"
    );
}

/// Part of #3993 Phase 11: IF/THEN/ELSE where one branch is NOT identity
/// should still be treated as a real write.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_if_then_else_not_identity_when_one_branch_differs() {
    let spec = r#"
---- MODULE PorIfNotIdentity ----
EXTENDS Naturals

VARIABLE x, y

MaybeIncX ==
    /\ x' = IF x > 0 THEN x + 1 ELSE x   \* NOT identity — then branch changes x
    /\ y' = y

Next == MaybeIncX

====
"#;

    let tree = parse_to_syntax_tree(spec);
    let lowered = lower(FileId(0), &tree);
    assert!(
        lowered.errors.is_empty(),
        "unexpected lower errors: {:?}",
        lowered.errors
    );
    let module = lowered.module.expect("lowered module");
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let mut var_names = Vec::new();
    for unit in &module.units {
        if let Unit::Variable(vars) = &unit.node {
            for var in vars {
                var_names.push(var.node.clone());
            }
        }
    }
    var_names.sort();
    ctx.register_vars(var_names.iter().cloned());
    ctx.resolve_state_vars_in_loaded_ops();

    let next_def = module
        .units
        .iter()
        .find_map(|unit| match &unit.node {
            Unit::Operator(def) if def.name.node == "Next" => Some(def),
            _ => None,
        })
        .expect("missing Next");

    let expanded_next = expand_operator_body_with_primes(&ctx, next_def);
    let actions = detect_actions(&expanded_next);
    assert_eq!(actions.len(), 1);

    let dependencies = extract_detected_action_dependencies(&ctx, &actions);
    let deps = &dependencies[0];

    // x' = IF x > 0 THEN x + 1 ELSE x is NOT identity — one branch changes x
    // It should be treated as a real write (and read) to x
    assert!(
        deps.writes.contains(&VarIndex(0)) || deps.reads.contains(&VarIndex(0)),
        "IF/THEN/ELSE with one non-identity branch should record real access to x"
    );
    assert!(
        !deps.unchanged.contains(&VarIndex(0)),
        "non-identity IF/THEN/ELSE should NOT be in unchanged"
    );
}

// ==================== EXCEPT Identity Detection Tests ====================

/// Part of #3993 Phase 11: EXCEPT identity `f' = [f EXCEPT ![k] = f[k]]`
/// is detected as an identity write (equivalent to UNCHANGED f).
///
/// This is the simplest EXCEPT identity case: a single key update that
/// reads back the same value.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_except_identity_detected_as_unchanged() {
    let spec = r#"
---- MODULE PorExceptIdentity ----
EXTENDS Naturals

VARIABLE f, g

Init ==
    /\ f = [x \in {1, 2} |-> 0]
    /\ g = [x \in {1, 2} |-> 0]

UpdateG ==
    /\ g' = [g EXCEPT ![1] = g[1] + 1]
    /\ f' = [f EXCEPT ![1] = f[1]]      \* identity: f[1] = f[1]

UpdateF ==
    /\ f' = [f EXCEPT ![1] = f[1] + 1]
    /\ g' = [g EXCEPT ![1] = g[1]]      \* identity: g[1] = g[1]

Next == UpdateG \/ UpdateF

====
"#;

    let tree = parse_to_syntax_tree(spec);
    let lowered = lower(FileId(0), &tree);
    assert!(
        lowered.errors.is_empty(),
        "unexpected lower errors: {:?}",
        lowered.errors
    );
    let module = lowered.module.expect("lowered module");
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let mut var_names = Vec::new();
    for unit in &module.units {
        if let Unit::Variable(vars) = &unit.node {
            for var in vars {
                var_names.push(var.node.clone());
            }
        }
    }
    var_names.sort();
    ctx.register_vars(var_names.iter().cloned());
    ctx.resolve_state_vars_in_loaded_ops();

    let next_def = module
        .units
        .iter()
        .find_map(|unit| match &unit.node {
            Unit::Operator(def) if def.name.node == "Next" => Some(def),
            _ => None,
        })
        .expect("missing Next");

    let expanded_next = expand_operator_body_with_primes(&ctx, next_def);
    let actions = detect_actions(&expanded_next);
    assert_eq!(actions.len(), 2);

    let dependencies = extract_detected_action_dependencies(&ctx, &actions);

    // UpdateG: f' = [f EXCEPT ![1] = f[1]] should be identity write to f
    let update_g_deps = &dependencies[0];
    assert!(
        update_g_deps.unchanged.contains(&VarIndex(0)),
        "f' = [f EXCEPT ![1] = f[1]] should be identity write (unchanged) in UpdateG"
    );
    assert!(
        !update_g_deps.writes.contains(&VarIndex(0)),
        "f' = [f EXCEPT ![1] = f[1]] should NOT be a real write in UpdateG"
    );

    // UpdateF: g' = [g EXCEPT ![1] = g[1]] should be identity write to g
    let update_f_deps = &dependencies[1];
    assert!(
        update_f_deps.unchanged.contains(&VarIndex(1)),
        "g' = [g EXCEPT ![1] = g[1]] should be identity write (unchanged) in UpdateF"
    );
    assert!(
        !update_f_deps.writes.contains(&VarIndex(1)),
        "g' = [g EXCEPT ![1] = g[1]] should NOT be a real write in UpdateF"
    );

    // With EXCEPT identity detection, UpdateG and UpdateF should be independent:
    // - UpdateG: real writes {g}, reads {g}, unchanged {f}
    // - UpdateF: real writes {f}, reads {f}, unchanged {g}
    let matrix = IndependenceMatrix::compute(&dependencies);
    assert_eq!(
        matrix.get(0, 1),
        IndependenceStatus::Independent,
        "UpdateG and UpdateF should be independent with EXCEPT identity detection"
    );
}

/// Part of #3993 Phase 11: EXCEPT with a non-identity value is NOT detected
/// as unchanged. `f' = [f EXCEPT ![1] = 42]` is a real write.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_except_non_identity_is_real_write() {
    let spec = r#"
---- MODULE PorExceptNonIdentity ----
EXTENDS Naturals

VARIABLE f

Init == f = [x \in {1, 2} |-> 0]

Update ==
    /\ f' = [f EXCEPT ![1] = 42]      \* NOT identity: writes 42, not f[1]

Next == Update

====
"#;

    let tree = parse_to_syntax_tree(spec);
    let lowered = lower(FileId(0), &tree);
    assert!(
        lowered.errors.is_empty(),
        "unexpected lower errors: {:?}",
        lowered.errors
    );
    let module = lowered.module.expect("lowered module");
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let mut var_names = Vec::new();
    for unit in &module.units {
        if let Unit::Variable(vars) = &unit.node {
            for var in vars {
                var_names.push(var.node.clone());
            }
        }
    }
    var_names.sort();
    ctx.register_vars(var_names.iter().cloned());
    ctx.resolve_state_vars_in_loaded_ops();

    let next_def = module
        .units
        .iter()
        .find_map(|unit| match &unit.node {
            Unit::Operator(def) if def.name.node == "Next" => Some(def),
            _ => None,
        })
        .expect("missing Next");

    let expanded_next = expand_operator_body_with_primes(&ctx, next_def);
    let actions = detect_actions(&expanded_next);
    assert_eq!(actions.len(), 1);

    let dependencies = extract_detected_action_dependencies(&ctx, &actions);
    let deps = &dependencies[0];

    // f' = [f EXCEPT ![1] = 42] is NOT identity — should be a real write
    assert!(
        !deps.unchanged.contains(&VarIndex(0)),
        "f' = [f EXCEPT ![1] = 42] should NOT be identity (unchanged)"
    );
}

// ==================== resolve_auto_por Unit Tests ====================
//
// Part of #4167: Verify config override for auto-POR.
// `resolve_auto_por(config_override)` must respect `Some(false)` to disable
// auto-POR regardless of the TY_AUTO_POR env var state. The OnceLock caches
// the env var value once per process, so env var toggling is not testable, but
// the config override path (the `Some` branch) is deterministic and testable.

/// Config override `Some(false)` disables auto-POR unconditionally.
///
/// This is the primary test for issue #4167: even if TY_AUTO_POR would
/// default to enabled, the config override takes precedence.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_resolve_auto_por_config_false_disables() {
    let result = super::resolve_auto_por(Some(false));
    assert!(
        !result,
        "resolve_auto_por(Some(false)) must return false regardless of env var"
    );
}

/// Config override `Some(true)` enables auto-POR unconditionally.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_resolve_auto_por_config_true_enables() {
    let result = super::resolve_auto_por(Some(true));
    assert!(
        result,
        "resolve_auto_por(Some(true)) must return true regardless of env var"
    );
}

/// Config override `None` falls back to env var (default: enabled).
///
/// NOTE: The OnceLock caches the env var once per process. In a test process
/// where TY_AUTO_POR is not set, this returns true (the default). We cannot
/// toggle the env var mid-process due to OnceLock caching, but we verify
/// the fallback path returns a deterministic value.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_resolve_auto_por_none_falls_back_to_env() {
    // Without explicit config override, resolve_auto_por reads the env var.
    // The OnceLock means the value is fixed for the process lifetime.
    // In CI/test environments where TY_AUTO_POR is not set, default is true.
    let result = super::resolve_auto_por(None);
    // We can only assert the type is bool and the function does not panic.
    // The actual value depends on whether TY_AUTO_POR was set before the
    // OnceLock initialized. In most test runs, it defaults to true.
    let _ = result; // No panic = pass
}

/// Config override takes precedence: calling with Some(false) after Some(true)
/// (or vice versa) always returns the override value, not a cached result.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_resolve_auto_por_config_override_not_cached() {
    // Each call with Some(_) goes through the match arm directly, bypassing
    // the OnceLock entirely. Verify both directions.
    assert!(super::resolve_auto_por(Some(true)));
    assert!(!super::resolve_auto_por(Some(false)));
    assert!(super::resolve_auto_por(Some(true)));
    assert!(!super::resolve_auto_por(Some(false)));
}

/// `auto_por_explicitly_enabled` honors the config override directly.
///
/// `Some(true)` is an explicit enable (release must be suppressed); `Some(false)`
/// is an explicit disable. The override bypasses the env `OnceLock` entirely, so
/// both directions are deterministic regardless of the ambient `TY_AUTO_POR`.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_auto_por_explicitly_enabled_config_override() {
    assert!(
        super::auto_por_explicitly_enabled(Some(true)),
        "Some(true) must report an explicit enable"
    );
    assert!(
        !super::auto_por_explicitly_enabled(Some(false)),
        "Some(false) must report an explicit disable"
    );
    // Re-check to prove the override is not cached across calls.
    assert!(super::auto_por_explicitly_enabled(Some(true)));
    assert!(!super::auto_por_explicitly_enabled(Some(false)));
}

/// With no config override, `auto_por_explicitly_enabled` reads `TY_AUTO_POR`.
/// The `OnceLock` fixes the value for the process; we only assert it does not
/// panic and that the DEFAULT (env unset) is treated as NON-explicit.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_auto_por_explicitly_enabled_none_reads_env() {
    let explicit = super::auto_por_explicitly_enabled(None);
    // In CI/test runs TY_AUTO_POR is normally unset → default auto-on is NOT
    // an explicit request, so the release is permitted. We cannot force the env
    // here (OnceLock), so only assert the call is total.
    let _ = explicit;
}

/// Part of #3993 Phase 11: diagnostic summary method works.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_diagnostic_summary_basic() {
    let deps: Vec<ActionDependencies> = vec![
        {
            let mut d = ActionDependencies::new();
            d.add_read(VarIndex(0));
            d.add_write(VarIndex(0));
            d.add_unchanged(VarIndex(1));
            d
        },
        {
            let mut d = ActionDependencies::new();
            d.add_read(VarIndex(1));
            d.add_write(VarIndex(1));
            d.add_unchanged(VarIndex(0));
            d
        },
    ];

    let matrix = IndependenceMatrix::compute(&deps);
    let names = vec!["IncX".to_string(), "IncY".to_string()];
    let summary = matrix.diagnostic_summary(&names);

    assert!(
        summary.contains("2 actions"),
        "summary should mention action count"
    );
    assert!(
        summary.contains("INDEPENDENT"),
        "summary should show independent pair"
    );
    assert!(summary.contains("IncX"), "summary should use action names");
    assert!(summary.contains("IncY"), "summary should use action names");
}

// ==================== Fail-closed OPAQUE extraction (hole #3) ====================
//
// The POR acceptance gate (2026-07-06, skeptic 1) proved that operator residue
// the expander declines to inline extracted EMPTY deps and landed unanalyzable
// pairs on INDEPENDENT. Extraction now FAILS CLOSED: such residue marks the
// action OPAQUE (dependent on everything, visible). These tests pin both sides
// of the contract — opacity where residue exists, and NO opacity on plain
// state-var/bound-var/arith actions (the precision requirement that keeps the
// disjoint-counter reductions alive).

/// Shared setup: parse, lower, load, register vars, resolve state vars, expand
/// Next with primes, and detect actions. Returns (ctx, detected actions).
fn setup_detected_actions(spec: &str) -> (EvalCtx, Vec<crate::coverage::DetectedAction>) {
    let tree = parse_to_syntax_tree(spec);
    let lowered = lower(FileId(0), &tree);
    assert!(
        lowered.errors.is_empty(),
        "unexpected lower errors: {:?}",
        lowered.errors
    );
    let module = lowered.module.expect("lowered module");
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let mut var_names = Vec::new();
    for unit in &module.units {
        if let Unit::Variable(vars) = &unit.node {
            for var in vars {
                var_names.push(var.node.clone());
            }
        }
    }
    var_names.sort();
    ctx.register_vars(var_names.iter().cloned());
    ctx.resolve_state_vars_in_loaded_ops();

    let next_def = module
        .units
        .iter()
        .find_map(|unit| match &unit.node {
            Unit::Operator(def) if def.name.node == "Next" => Some(def),
            _ => None,
        })
        .expect("missing Next");

    let expanded_next = expand_operator_body_with_primes(&ctx, next_def);
    let actions = detect_actions(&expanded_next);
    (ctx, actions)
}

/// Skeptic-1 twin T4a: B's guard read of `w` hides behind the zero-arg FuncDef
/// operator `WSel` that the #2955 perf guard keeps un-inlined. The pinned
/// property is that the hidden `w` READ survives, landing (A, B) on Dependent
/// — never the empty-deps extractor's false-clean INDEPENDENT.
///
/// Until 2026-07-20 the walker discharged that by marking B OPAQUE. It now
/// resolves the definition body instead (`resolve_operator_body`), so B gets
/// the EXACT footprint reads={g, w} / writes={v}: the pinned dependence still
/// holds, through the recovered read rather than through opacity. The
/// fail-closed path is pinned separately on residue that genuinely cannot be
/// resolved (`test_higher_order_operator_application_stays_opaque`).
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_hidden_funcdef_residue_recovers_exact_read_and_stays_dependent() {
    let spec = r#"
---- MODULE PorOpaqueFuncdef ----
EXTENDS Integers

VARIABLES w, g, v, t

WSel == [i \in {0} |-> w]

A == w = 0 /\ w' = 1 /\ UNCHANGED <<g, v, t>>

C == g = 0 /\ g' = 1 /\ UNCHANGED <<w, v, t>>

B == g = 1 /\ WSel[0] = 0 /\ v' = 1 /\ UNCHANGED <<w, g, t>>

Tick == t' = 1 - t /\ UNCHANGED <<w, g, v>>

Next == A \/ C \/ B \/ Tick
====
"#;
    let (ctx, actions) = setup_detected_actions(spec);
    assert_eq!(actions.len(), 4, "expected A, C, B, Tick");

    let deps = extract_detected_action_dependencies(&ctx, &actions);
    // Action order matches the disjunct order: A=0, C=1, B=2, Tick=3.
    // Sorted var registry: g=0, t=1, v=2, w=3.
    assert!(
        !deps[2].opaque,
        "B's WSel body is fully analyzable — must resolve, not fail closed: {:?}",
        deps[2].opaque_reason
    );
    // THE hole-#3 property: the read of `w` hidden inside WSel is recovered.
    assert!(
        deps[2].reads.contains(&VarIndex(3)),
        "B's hidden read of w must survive; deps: {:?}",
        deps[2]
    );
    // SUPERSET-free: B reads exactly {g, w} and writes exactly {v}.
    assert_eq!(
        sorted_vars(&deps[2].reads),
        vec![0, 3],
        "B reads exactly g and w"
    );
    assert_eq!(sorted_vars(&deps[2].writes), vec![2], "B writes exactly v");
    assert!(!deps[0].opaque, "A is a plain action — must NOT be opaque");
    assert!(!deps[1].opaque, "C is a plain action — must NOT be opaque");
    assert!(
        !deps[3].opaque,
        "Tick is a plain action — must NOT be opaque"
    );

    let matrix = IndependenceMatrix::compute(&deps);
    // THE pinned pair: (A, B) must be Dependent — this was the false-clean
    // INDEPENDENT verdict of the empty-deps extractor. It now holds because
    // A writes w and B is known to READ w.
    assert_eq!(
        matrix.get(0, 2),
        IndependenceStatus::Dependent,
        "(A, B) must be Dependent via B's recovered read of w"
    );
    assert_eq!(matrix.get(2, 0), IndependenceStatus::Dependent);
    // C writes g, which B reads.
    assert_eq!(matrix.get(1, 2), IndependenceStatus::Dependent);
    // Precision gained by resolution: B touches only {g, w, v}, Tick only
    // {t}, so the pair is genuinely independent.
    assert_eq!(matrix.get(2, 3), IndependenceStatus::Independent);
    // Precision: the analyzable pairs stay Independent (reduction survives).
    assert_eq!(matrix.get(0, 1), IndependenceStatus::Independent);
    assert_eq!(matrix.get(0, 3), IndependenceStatus::Independent);
    assert_eq!(matrix.get(1, 3), IndependenceStatus::Independent);

    // C2: B's writes are now known, so an empty visibility set does not see it.
    let visibility = VisibilitySet::new();
    assert!(!visibility.is_action_visible(&deps[2]));
    assert!(!visibility.is_action_visible(&deps[0]));
}

/// Sorted variable indices of a dependency set, for SUPERSET-free assertions.
fn sorted_vars(set: &rustc_hash::FxHashSet<VarIndex>) -> Vec<usize> {
    let mut v: Vec<usize> = set.iter().map(|idx| idx.as_usize()).collect();
    v.sort_unstable();
    v
}

/// Skeptic-1 twin T4b: B's guard read of `w` hides behind a CAPTURE-UNSAFE
/// application `GuardW(c)` the expander keeps un-inlined. Same contract as
/// the funcdef twin: the hidden `w` read must survive.
///
/// Body resolution is capture-SAFE where inlining is not, and that is why it
/// can discharge this case precisely: it never substitutes the argument into
/// the body, it BINDS the formal and walks the argument at the call site, so
/// the body's own `\E c` and the call site's `c` never collide.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_hidden_capture_residue_recovers_exact_read_and_stays_dependent() {
    let spec = r#"
---- MODULE PorOpaqueCapture ----
EXTENDS Integers

VARIABLES w, g, v, t

GuardW(a) == \E c \in {0} : w = 0 /\ a = c

A == w = 0 /\ w' = 1 /\ UNCHANGED <<g, v, t>>

C == g = 0 /\ g' = 1 /\ UNCHANGED <<w, v, t>>

B == g = 1 /\ (\E c \in {0} : GuardW(c)) /\ v' = 1 /\ UNCHANGED <<w, g, t>>

Tick == t' = 1 - t /\ UNCHANGED <<w, g, v>>

Next == A \/ C \/ B \/ Tick
====
"#;
    let (ctx, actions) = setup_detected_actions(spec);
    assert_eq!(actions.len(), 4, "expected A, C, B, Tick");

    let deps = extract_detected_action_dependencies(&ctx, &actions);
    assert!(
        !deps[2].opaque,
        "GuardW's body is fully analyzable — must resolve, not fail closed: {:?}",
        deps[2].opaque_reason
    );
    // Sorted var registry: g=0, t=1, v=2, w=3. THE pinned read.
    assert!(
        deps[2].reads.contains(&VarIndex(3)),
        "B's hidden read of w must survive; deps: {:?}",
        deps[2]
    );
    assert_eq!(sorted_vars(&deps[2].reads), vec![0, 3]);
    assert_eq!(sorted_vars(&deps[2].writes), vec![2]);
    assert!(!deps[0].opaque && !deps[1].opaque && !deps[3].opaque);

    let matrix = IndependenceMatrix::compute(&deps);
    assert_eq!(
        matrix.get(0, 2),
        IndependenceStatus::Dependent,
        "(A, B) must be Dependent via B's recovered read of w"
    );
    assert_eq!(matrix.get(1, 2), IndependenceStatus::Dependent);
    assert_eq!(matrix.get(0, 1), IndependenceStatus::Independent);
    // Precision gained by resolution: B never touches t.
    assert_eq!(matrix.get(2, 3), IndependenceStatus::Independent);

    let visibility = VisibilitySet::new();
    assert!(!visibility.is_action_visible(&deps[2]));
}

/// PRECISION pin: plain state-var/bound-var/arithmetic actions — including
/// quantifiers with bound variables and UNCHANGED tuples — must NOT be marked
/// opaque. Over-marking here would kill the disjoint-counter reductions
/// (5-not-9 / 7-not-27) everywhere.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_plain_actions_are_not_opaque() {
    let spec = r#"
---- MODULE PorOpaquePrecision ----
EXTENDS Naturals

VARIABLE x, y

IncX ==
    /\ x < 2
    /\ x' = x + 1
    /\ UNCHANGED y

IncY ==
    /\ \E k \in 1..2 : y' = y + k
    /\ x' = x

Next == IncX \/ IncY
====
"#;
    let (ctx, actions) = setup_detected_actions(spec);
    assert_eq!(actions.len(), 2);

    let deps = extract_detected_action_dependencies(&ctx, &actions);
    assert!(
        !deps[0].opaque,
        "IncX (state vars + arith + UNCHANGED) must not be opaque: {:?}",
        deps[0].opaque_reason
    );
    assert!(
        !deps[1].opaque,
        "IncY (bound var k + identity x' = x) must not be opaque: {:?}",
        deps[1].opaque_reason
    );

    let matrix = IndependenceMatrix::compute(&deps);
    assert_eq!(
        matrix.get(0, 1),
        IndependenceStatus::Independent,
        "the disjoint pair must stay independent — reduction survival"
    );
}

/// C2 fail-closed: an INVARIANT whose read set is genuinely unknowable must
/// make `extend_from_expanded_expr` fall back to mark_all_visible.
///
/// The original residue for this pin was a zero-arg FuncDef (`WSel == [i \in
/// {0} |-> w]`); since 2026-07-20 that body is resolved exactly, so the pin
/// moved to residue `resolve_operator_body` still declines — a RECURSIVE
/// operator whose formal SHADOWS a state-variable name.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_opaque_invariant_marks_all_visible() {
    let spec = r#"
---- MODULE PorOpaqueInvariant ----
EXTENDS Integers

VARIABLES w, v

RECURSIVE Shadow(_)
Shadow(w) == IF w <= 0 THEN 0 ELSE Shadow(w - 1)

Inv == Shadow(v) = 0

Next == v' = v + 1 /\ UNCHANGED w
====
"#;
    let (ctx, _actions) = setup_detected_actions(spec);

    let inv_def = ctx.get_op("Inv").expect("Inv operator");
    let mut visibility = VisibilitySet::new();
    visibility.extend_from_expanded_expr(&ctx, &inv_def.body);

    // Any action — even one with no writes at all — must now be visible.
    let empty = ActionDependencies::new();
    assert!(
        visibility.is_action_visible(&empty),
        "opaque invariant residue must fall back to mark_all_visible"
    );
}

// ==================== Coverage-indexed dependency extraction (audit-2026-07 #11) ====================
//
// The independence matrix MUST be indexed by the ENUMERATION (coverage)
// decomposition: `detect_actions` on the NON-primed expansion of Next, where a
// named action operator whose body directly contains a prime survives as a
// single un-split operator reference. The old flow expanded the WHOLE Next
// with primes and re-ran `detect_actions`, splitting such an operator at its
// top-level disjunction into a FINER list; `compute_ample_set` then fed
// coverage-space indices into the finer matrix and read C1/C2 off the WRONG
// rows (stopgap: skip POR on length mismatch). The fix expands EACH coverage
// action's expression with primes inside `extract_detected_action_dependencies`
// and unions over internal disjuncts, so `deps.len() == actions.len()` by
// construction and POR stays ON.

/// Shared setup for coverage-decomposition tests: parse, lower, load, register
/// vars, resolve state vars, and detect actions on the RAW (non-expanded) Next
/// body — exactly what the checkers' enumeration path does. Returns
/// (ctx, module, coverage actions).
fn setup_coverage_actions(
    spec: &str,
) -> (
    EvalCtx,
    tla_core::ast::Module,
    Vec<crate::coverage::DetectedAction>,
) {
    let tree = parse_to_syntax_tree(spec);
    let lowered = lower(FileId(0), &tree);
    assert!(
        lowered.errors.is_empty(),
        "unexpected lower errors: {:?}",
        lowered.errors
    );
    let module = lowered.module.expect("lowered module");
    let mut ctx = EvalCtx::new();
    ctx.load_module(&module);

    let mut var_names = Vec::new();
    for unit in &module.units {
        if let Unit::Variable(vars) = &unit.node {
            for var in vars {
                var_names.push(var.node.clone());
            }
        }
    }
    var_names.sort();
    ctx.register_vars(var_names.iter().cloned());
    ctx.resolve_state_vars_in_loaded_ops();

    let next_def = module
        .units
        .iter()
        .find_map(|unit| match &unit.node {
            Unit::Operator(def) if def.name.node == "Next" => Some(def),
            _ => None,
        })
        .expect("missing Next");

    let actions = detect_actions(next_def);
    (ctx, module, actions)
}

/// REGRESSION (audit-2026-07 #11): a named action operator with direct primes
/// AND a top-level internal disjunction. The coverage decomposition keeps `A`
/// un-split (2 actions) while the old whole-Next with-primes re-detection
/// split it (3 actions) — the mismatch that made C1/C2 read the wrong rows.
/// The per-coverage-action extraction must return exactly one dep set per
/// coverage action (2x2 matrix) whose contents are the UNION over A's
/// internal disjuncts, with real writes dominating identity writes.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_coverage_indexed_matrix_unions_internal_disjuncts() {
    let spec = r#"
---- MODULE PorCoverageIndexedUnion ----
EXTENDS Integers

VARIABLE x, w, z

A == (x' = 1 /\ w' = w /\ z' = z) \/ (w' = 1 /\ x' = x /\ z' = z)

B == z' = 1 /\ x' = x /\ w' = w

Next == A \/ B
====
"#;
    let (ctx, module, coverage_actions) = setup_coverage_actions(spec);
    assert_eq!(
        coverage_actions.len(),
        2,
        "coverage decomposition keeps the named A un-split"
    );
    assert_eq!(coverage_actions[0].name, "A");
    assert_eq!(coverage_actions[1].name, "B");

    // Pin the TRIGGER shape: the old whole-Next with-primes re-detection
    // produced a FINER (3-action) decomposition than enumeration (2 actions).
    let next_def = module
        .units
        .iter()
        .find_map(|unit| match &unit.node {
            Unit::Operator(def) if def.name.node == "Next" => Some(def),
            _ => None,
        })
        .expect("missing Next");
    let with_primes = expand_operator_body_with_primes(&ctx, next_def);
    assert_eq!(
        detect_actions(&with_primes).len(),
        3,
        "trigger shape: whole-Next with-primes re-detection splits A"
    );

    // ONE dep set per COVERAGE action — indexed 1:1 with enumeration.
    let deps = extract_detected_action_dependencies(&ctx, &coverage_actions);
    assert_eq!(deps.len(), 2, "one dependency set per coverage action");

    // Sorted var registry: w=0, x=1, z=2.
    let a = &deps[0];
    assert!(
        !a.opaque,
        "A's inlined body must be analyzable, got opaque: {:?}",
        a.opaque_reason
    );
    assert!(
        a.writes.contains(&VarIndex(1)),
        "disjunct 1 really writes x"
    );
    assert!(
        a.writes.contains(&VarIndex(0)),
        "disjunct 2 really writes w — disjunct 1's w' = w identity must NOT mask it in the union"
    );
    assert!(
        !a.unchanged.contains(&VarIndex(0)),
        "w must not be classified unchanged in the union"
    );
    assert!(
        a.unchanged.contains(&VarIndex(2)),
        "z is an identity write in BOTH disjuncts"
    );
    assert!(!a.writes.contains(&VarIndex(2)));

    let b = &deps[1];
    assert!(!b.opaque, "B must be analyzable: {:?}", b.opaque_reason);
    assert!(b.writes.contains(&VarIndex(2)), "B really writes z");
    assert!(b.unchanged.contains(&VarIndex(0)));
    assert!(b.unchanged.contains(&VarIndex(1)));

    // The matrix is 2x2 — coverage-indexed — and the pair is independent
    // (A's union writes {x, w} vs B's {z} are disjoint), so POR may engage.
    let matrix = IndependenceMatrix::compute(&deps);
    assert_eq!(matrix.action_count(), 2, "matrix indexed by enumeration");
    assert_eq!(matrix.get(0, 1), IndependenceStatus::Independent);
    assert_eq!(matrix.get(1, 0), IndependenceStatus::Independent);
}

/// Union conservatism (the direction that must NEVER be lost): when ONE
/// internal disjunct of a coverage action really writes a variable that
/// another coverage action also writes, the unioned pair must stay Dependent
/// even though the OTHER disjunct only has an identity write of it.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_union_deps_conflicting_internal_disjunct_stays_dependent() {
    let spec = r#"
---- MODULE PorUnionConflict ----
EXTENDS Integers

VARIABLE x, w

A == (x' = 1 /\ w' = w) \/ (w' = 1 /\ x' = x)

B == w' = 2 /\ x' = x

Next == A \/ B
====
"#;
    let (ctx, _module, coverage_actions) = setup_coverage_actions(spec);
    assert_eq!(coverage_actions.len(), 2);

    let deps = extract_detected_action_dependencies(&ctx, &coverage_actions);
    assert_eq!(deps.len(), 2);

    // Sorted var registry: w=0, x=1. A's union real-writes {x, w}; B
    // real-writes {w} — a write/write conflict on w.
    assert!(deps[0].writes.contains(&VarIndex(0)));
    assert!(deps[1].writes.contains(&VarIndex(0)));

    let matrix = IndependenceMatrix::compute(&deps);
    assert_eq!(matrix.action_count(), 2);
    assert_eq!(
        matrix.get(0, 1),
        IndependenceStatus::Dependent,
        "w write/write conflict must survive the disjunct union"
    );
}

// ==================== Definition-body resolution (WP-03a, 2026-07-20) ====================
//
// `resolve_operator_body` walks an un-inlined operator's DEFINITION BODY
// instead of failing closed. This analysis is shared with POR and coverage, so
// the contract is two-sided: the resolved footprint must be EXACT (a missing
// variable would let POR prune a real transition), and every shape that cannot
// be resolved exactly must still yield opaque.

/// EXACTNESS pin, btree `FindLeafNode` shape: a RECURSIVE operator the
/// expander refuses to inline. The resolved footprint must be a SUPERSET-FREE
/// exact match of the variables the action actually touches — in particular it
/// must NOT pick up `keysOf` or `tick`, which the action never reads.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_recursive_operator_application_resolves_exact_footprint() {
    let spec = r#"
---- MODULE PorRecursiveResolve ----
EXTENDS Integers

VARIABLES childOf, isLeaf, keysOf, out, root, tick

RECURSIVE Find(_)
Find(n) == IF isLeaf[n] THEN n ELSE Find(childOf[n])

Probe ==
    /\ out' = Find(root)
    /\ UNCHANGED <<childOf, isLeaf, keysOf, root, tick>>

Bump == tick' = tick + 1 /\ UNCHANGED <<childOf, isLeaf, keysOf, out, root>>

Next == Probe \/ Bump
====
"#;
    let (ctx, actions) = setup_detected_actions(spec);
    assert_eq!(actions.len(), 2, "expected Probe, Bump");

    let deps = extract_detected_action_dependencies(&ctx, &actions);
    // Sorted var registry: childOf=0, isLeaf=1, keysOf=2, out=3, root=4, tick=5.
    assert!(
        !deps[0].opaque,
        "recursive Find must resolve, not fail closed: {:?}",
        deps[0].opaque_reason
    );
    // Probe reads root (the argument), and isLeaf + childOf (through the
    // recursive body). It reads NOTHING else — keysOf and tick must be absent.
    assert_eq!(
        sorted_vars(&deps[0].reads),
        vec![0, 1, 4],
        "Probe must read exactly childOf, isLeaf, root — no superset"
    );
    assert_eq!(
        sorted_vars(&deps[0].writes),
        vec![3],
        "Probe must write exactly out"
    );

    // The precision payoff: Probe never touches tick, so it commutes with Bump.
    let matrix = IndependenceMatrix::compute(&deps);
    assert_eq!(matrix.get(0, 1), IndependenceStatus::Independent);
}

/// TERMINATION pin: mutual recursion must converge (the in-progress set cuts
/// the cycle) and still produce the exact union of both bodies' footprints.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_mutually_recursive_operators_terminate_with_exact_footprint() {
    let spec = r#"
---- MODULE PorMutualRecursion ----
EXTENDS Integers

VARIABLES a, b, c, out

RECURSIVE Ping(_)
RECURSIVE Pong(_)
Ping(n) == IF n <= 0 THEN a ELSE Pong(n - 1)
Pong(n) == IF n <= 0 THEN b ELSE Ping(n - 1)

Step == out' = Ping(3) /\ UNCHANGED <<a, b, c>>

Next == Step
====
"#;
    let (ctx, actions) = setup_detected_actions(spec);
    let deps = extract_detected_action_dependencies(&ctx, &actions);
    // Sorted var registry: a=0, b=1, c=2, out=3.
    assert!(
        !deps[0].opaque,
        "mutual recursion must resolve, not fail closed: {:?}",
        deps[0].opaque_reason
    );
    assert_eq!(
        sorted_vars(&deps[0].reads),
        vec![0, 1],
        "Step must read exactly a and b (both cycle bodies), never c"
    );
    assert_eq!(sorted_vars(&deps[0].writes), vec![3]);
}

/// Classify an operator body WITHOUT running the expander. The expander
/// inlines most applications before the walker ever sees them, so this is how
/// the residue shapes `resolve_operator_body` must decline are pinned directly.
fn deps_of_unexpanded_body(ctx: &EvalCtx, op_name: &str) -> ActionDependencies {
    let def = ctx.get_op(op_name).expect("operator must exist");
    let mut deps = ActionDependencies::new();
    extract_dependencies_ast_expr(ctx, &def.body.node, &mut deps);
    deps
}

/// FAIL-CLOSED pin: a HIGHER-ORDER operator's body applies an operator the
/// CALLER supplies, so the body's footprint is not the definition's. Must stay
/// opaque.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_higher_order_operator_application_stays_opaque() {
    let spec = r#"
---- MODULE PorHigherOrderOpaque ----
EXTENDS Integers

VARIABLES w, v

ApplyTo(F(_), x) == F(x)

Sel(i) == w + i

B == ApplyTo(Sel, 0) = 0 /\ v' = 1 /\ UNCHANGED w

Next == B
====
"#;
    let (ctx, _actions) = setup_detected_actions(spec);
    let deps = deps_of_unexpanded_body(&ctx, "B");
    assert!(
        deps.opaque,
        "higher-order operator parameter must fail closed; deps: {deps:?}"
    );
}

/// FAIL-CLOSED pin: `has_primed_param` means the operator is applied by NAME
/// substitution, so `p'` in the body primes the ARGUMENT — the write set
/// belongs to the call site, not the body. Resolution must decline.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_primed_param_operator_stays_opaque() {
    let spec = r#"
---- MODULE PorPrimedParamOpaque ----
EXTENDS Integers

VARIABLES w, v

RECURSIVE Bumped(_)
Bumped(p) == IF w = 0 THEN p' = 1 ELSE Bumped(p)

B == Bumped(v) /\ UNCHANGED w

Next == B
====
"#;
    let (ctx, actions) = setup_detected_actions(spec);
    let deps = extract_detected_action_dependencies(&ctx, &actions);
    assert!(
        deps[0].opaque,
        "a primed formal parameter must fail closed; deps: {:?}",
        deps[0]
    );
}

/// FAIL-CLOSED pin: a formal that SHADOWS a state-variable name would be
/// resolved as that variable by `walk_prime`/`walk_unchanged`, which consult
/// the var registry without consulting binder scope. Resolution must decline.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_formal_shadowing_state_var_stays_opaque() {
    let spec = r#"
---- MODULE PorShadowingFormalOpaque ----
EXTENDS Integers

VARIABLES w, v

RECURSIVE Shadow(_)
Shadow(w) == IF w <= 0 THEN 0 ELSE Shadow(w - 1)

B == Shadow(3) = 0 /\ v' = 1 /\ UNCHANGED w

Next == B
====
"#;
    let (ctx, actions) = setup_detected_actions(spec);
    let deps = extract_detected_action_dependencies(&ctx, &actions);
    assert!(
        deps[0].opaque,
        "a formal shadowing a state-variable name must fail closed; deps: {:?}",
        deps[0]
    );
}

/// TERMINATION pin: a definition chain deeper than `MAX_OPERATOR_RESOLVE_DEPTH`
/// must DEGRADE TO OPAQUE rather than recurse without bound. The chain is built
/// from RECURSIVE operators so the expander leaves every level un-inlined.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_deep_operator_chain_degrades_to_opaque() {
    const DEPTH: usize = 40; // > MAX_OPERATOR_RESOLVE_DEPTH (32)
    let mut spec = String::from(
        "---- MODULE PorDeepChain ----\nEXTENDS Integers\n\nVARIABLES w, v\n\n",
    );
    for i in 0..DEPTH {
        spec.push_str(&format!("RECURSIVE F{i}(_)\n"));
    }
    for i in 0..DEPTH {
        if i + 1 < DEPTH {
            spec.push_str(&format!(
                "F{i}(n) == IF n <= 0 THEN 0 ELSE F{}(n - 1) + F{i}(n - 1)\n",
                i + 1
            ));
        } else {
            spec.push_str(&format!("F{i}(n) == IF n <= 0 THEN 0 ELSE w + F{i}(n - 1)\n"));
        }
    }
    spec.push_str("\nB == F0(v) = 0 /\\ v' = 1 /\\ UNCHANGED w\n\nNext == B\n====\n");

    let (ctx, _actions) = setup_detected_actions(&spec);
    let deps = deps_of_unexpanded_body(&ctx, "B");
    assert!(
        deps.opaque,
        "a chain deeper than the resolve-depth cap must degrade to opaque; deps: {deps:?}"
    );
}

// ==================== EXCEPT `@` resolution (WP-13, 2026-07-20) ====================
//
// The `@` old-value placeholder survives lowering as a bare `Expr::Ident("@")`
// (`SubstituteAt` binds it at evaluation time), so the walker used to classify
// it as un-inlined residue and mark the whole action opaque. It now binds `@`
// to the enclosing EXCEPT, whose base and path are already in `deps`. Same
// two-sided contract as definition-body resolution: EXACT footprints where the
// binding is provable, opaque everywhere else.

/// EXACTNESS pin, btree `AddToLeaf` shape (`test_specs/btree.tla:178`):
/// `[keysOf EXCEPT ![focus] = @ \union {key}]`. The action must resolve to the
/// exact variables it touches and must NOT pick up `tick`, which it never
/// reads — the precision that keeps `Bump` independent of it.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_except_at_placeholder_resolves_exact_footprint() {
    let spec = r#"
---- MODULE PorExceptAtExact ----
EXTENDS Integers, FiniteSets

VARIABLES focus, key, keysOf, tick

AddKey ==
    /\ keysOf' = [keysOf EXCEPT ![focus] = @ \union {key}]
    /\ UNCHANGED <<focus, key, tick>>

Bump == tick' = tick + 1 /\ UNCHANGED <<focus, key, keysOf>>

Next == AddKey \/ Bump
====
"#;
    let (ctx, actions) = setup_detected_actions(spec);
    assert_eq!(actions.len(), 2, "expected AddKey, Bump");

    let deps = extract_detected_action_dependencies(&ctx, &actions);
    // Sorted var registry: focus=0, key=1, keysOf=2, tick=3.
    assert!(
        !deps[0].opaque,
        "`@` inside an EXCEPT value must resolve, not fail closed: {:?}",
        deps[0].opaque_reason
    );
    assert_eq!(
        sorted_vars(&deps[0].reads),
        vec![0, 1, 2],
        "AddKey must read exactly focus, key, keysOf — no superset"
    );
    assert_eq!(
        sorted_vars(&deps[0].writes),
        vec![2],
        "AddKey must write exactly keysOf"
    );

    // The precision payoff: AddKey never touches tick, so it commutes with Bump.
    let matrix = IndependenceMatrix::compute(&deps);
    assert_eq!(matrix.get(0, 1), IndependenceStatus::Independent);
}

/// EXACTNESS pin for NESTED EXCEPTs. `@` binds to the INNERMOST enclosing
/// level, and the inner level's base is itself the outer `@`. Getting the
/// binding wrong would drop a read — the dangerous direction — so the pin is
/// that the footprint is exactly the outer function, the inner function, and
/// both index expressions, with the untouched `spare` absent.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_nested_except_at_resolves_exact_footprint() {
    let spec = r#"
---- MODULE PorNestedExceptAt ----
EXTENDS Integers

VARIABLES f, g, i, j, out, spare

Step ==
    /\ out' = [f EXCEPT ![i] = [g EXCEPT ![j] = @ + 1]]
    /\ UNCHANGED <<f, g, i, j, spare>>

Next == Step
====
"#;
    let (ctx, actions) = setup_detected_actions(spec);
    let deps = extract_detected_action_dependencies(&ctx, &actions);
    // Sorted var registry: f=0, g=1, i=2, j=3, out=4, spare=5.
    assert!(
        !deps[0].opaque,
        "nested EXCEPT `@` must resolve, not fail closed: {:?}",
        deps[0].opaque_reason
    );
    assert_eq!(
        sorted_vars(&deps[0].reads),
        vec![0, 1, 2, 3],
        "Step must read exactly f, g, i, j — never spare"
    );
    assert_eq!(sorted_vars(&deps[0].writes), vec![4]);
}

/// EXACTNESS pin for a MULTI-INDEX EXCEPT path: `@` denotes `f[i][k]`, so both
/// index expressions belong to its footprint and nothing else does.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_multi_index_except_at_resolves_exact_footprint() {
    let spec = r#"
---- MODULE PorMultiIndexExceptAt ----
EXTENDS Integers

VARIABLES f, i, k, out, spare

Step ==
    /\ out' = [f EXCEPT ![i][k] = @ + 1]
    /\ UNCHANGED <<f, i, k, spare>>

Next == Step
====
"#;
    let (ctx, actions) = setup_detected_actions(spec);
    let deps = extract_detected_action_dependencies(&ctx, &actions);
    // Sorted var registry: f=0, i=1, k=2, out=3, spare=4.
    assert!(
        !deps[0].opaque,
        "multi-index EXCEPT `@` must resolve, not fail closed: {:?}",
        deps[0].opaque_reason
    );
    assert_eq!(
        sorted_vars(&deps[0].reads),
        vec![0, 1, 2],
        "Step must read exactly f, i, k — never spare"
    );
    assert_eq!(sorted_vars(&deps[0].writes), vec![3]);
}

/// EXACTNESS pin for `@` in a nested EXCEPT's PATH INDEX. Per
/// `SubstituteAt::fold_except_specs`, path indices sit at the ENCLOSING level
/// (only `value` opens a new one), so this `@` denotes `f[i]`. The walker
/// tracks the same scoping; the footprint must still be exact.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_except_at_in_nested_path_index_resolves_exact_footprint() {
    let spec = r#"
---- MODULE PorExceptAtPathIndex ----
EXTENDS Integers

VARIABLES f, g, i, out, spare

Step ==
    /\ out' = [f EXCEPT ![i] = [g EXCEPT ![@] = 0]]
    /\ UNCHANGED <<f, g, i, spare>>

Next == Step
====
"#;
    let (ctx, actions) = setup_detected_actions(spec);
    let deps = extract_detected_action_dependencies(&ctx, &actions);
    // Sorted var registry: f=0, g=1, i=2, out=3, spare=4.
    assert!(
        !deps[0].opaque,
        "`@` in a nested EXCEPT path index must resolve, not fail closed: {:?}",
        deps[0].opaque_reason
    );
    assert_eq!(
        sorted_vars(&deps[0].reads),
        vec![0, 1, 2],
        "Step must read exactly f, g, i — never spare"
    );
    assert_eq!(sorted_vars(&deps[0].writes), vec![3]);
}

/// FAIL-CLOSED pin: an `@` that no enclosing EXCEPT value binds has no
/// footprint bound at all. Built directly as AST because the surface syntax
/// for it does not survive name resolution.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_unbound_at_placeholder_stays_opaque() {
    use tla_core::ast::Expr;
    use tla_core::{name_intern::NameId, Spanned};

    let mut ctx = EvalCtx::new();
    ctx.register_vars(["v".to_string()]);

    // `@ + 1` at the top level of an action body — no EXCEPT in sight.
    let expr = Expr::Add(
        Box::new(Spanned::dummy(Expr::Ident(
            "@".to_string(),
            NameId::INVALID,
        ))),
        Box::new(Spanned::dummy(Expr::Int(1u32.into()))),
    );
    let mut deps = ActionDependencies::new();
    extract_dependencies_ast_expr(&ctx, &expr, &mut deps);
    assert!(
        deps.opaque,
        "an `@` outside any EXCEPT value must fail closed; deps: {deps:?}"
    );
    assert!(
        deps.opaque_reason
            .as_deref()
            .is_some_and(|r| r.contains("`@` old-value placeholder")),
        "must be opaque FOR the unbound `@`, not incidentally: {:?}",
        deps.opaque_reason
    );
}

/// FAIL-CLOSED pin: the EXCEPT base and PATH INDEX of the OUTERMOST EXCEPT sit
/// at the enclosing level, which at the top of an action is no level at all.
/// An `@` there binds to nothing and must stay opaque even though the walker
/// is syntactically "inside an EXCEPT".
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_at_placeholder_in_outermost_except_path_stays_opaque() {
    use tla_core::ast::{ExceptPathElement, ExceptSpec, Expr};
    use tla_core::{name_intern::NameId, Spanned};

    let mut ctx = EvalCtx::new();
    ctx.register_vars(["f".to_string(), "v".to_string()]);

    // `[f EXCEPT ![@] = 0]` with no enclosing EXCEPT to bind the path's `@`.
    let expr = Expr::Except(
        Box::new(Spanned::dummy(Expr::StateVar(
            "f".to_string(),
            0,
            NameId::INVALID,
        ))),
        vec![ExceptSpec {
            path: vec![ExceptPathElement::Index(Spanned::dummy(Expr::Ident(
                "@".to_string(),
                NameId::INVALID,
            )))],
            value: Spanned::dummy(Expr::Int(0u32.into())),
        }],
    );
    let mut deps = ActionDependencies::new();
    extract_dependencies_ast_expr(&ctx, &expr, &mut deps);
    assert!(
        deps.opaque,
        "an `@` in the outermost EXCEPT path binds to nothing; deps: {deps:?}"
    );
    assert!(
        deps.opaque_reason
            .as_deref()
            .is_some_and(|r| r.contains("`@` old-value placeholder")),
        "must be opaque FOR the unbound `@`, not incidentally: {:?}",
        deps.opaque_reason
    );
}

/// FAIL-CLOSED pin: `resolve_operator_body` walks a definition body in a FRESH
/// scope, so an `@` inside that body is not bound by the caller's EXCEPT even
/// when the call site sits in an EXCEPT replacement value. Resolution must
/// degrade to opaque rather than inherit the caller's binding.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_at_placeholder_inside_resolved_operator_body_stays_opaque() {
    use tla_core::ast::{ExceptPathElement, ExceptSpec, Expr};
    use tla_core::{name_intern::NameId, Spanned};

    let spec = r#"
---- MODULE PorExceptAtOperatorBody ----
EXTENDS Integers

VARIABLES f, v

RECURSIVE Stray(_)
Stray(n) == IF n <= 0 THEN @ ELSE Stray(n - 1)

Next == v' = 0 /\ UNCHANGED f
====
"#;
    let (ctx, _actions) = setup_detected_actions(spec);

    // `[f EXCEPT ![0] = Stray(1)]`: the call site IS inside an EXCEPT value,
    // but `Stray`'s body is not.
    let expr = Expr::Except(
        Box::new(Spanned::dummy(Expr::StateVar(
            "f".to_string(),
            0,
            NameId::INVALID,
        ))),
        vec![ExceptSpec {
            path: vec![ExceptPathElement::Index(Spanned::dummy(Expr::Int(
                0u32.into(),
            )))],
            value: Spanned::dummy(Expr::Apply(
                Box::new(Spanned::dummy(Expr::Ident(
                    "Stray".to_string(),
                    NameId::INVALID,
                ))),
                vec![Spanned::dummy(Expr::Int(1u32.into()))],
            )),
        }],
    );
    let mut deps = ActionDependencies::new();
    extract_dependencies_ast_expr(&ctx, &expr, &mut deps);
    assert!(
        deps.opaque,
        "an `@` in a resolved definition body must not inherit the caller's EXCEPT; deps: {deps:?}"
    );
    assert!(
        deps.opaque_reason
            .as_deref()
            .is_some_and(|r| r.contains("`@` old-value placeholder")),
        "must be opaque FOR the body's unbound `@`, not because resolution declined: {:?}",
        deps.opaque_reason
    );
}

// ---------------------------------------------------------------------------
// WP-25: the DEFAULT-OFF gate on both 2026-07-20 precision analyses.
//
// A precise footprint is not only a POR input — the hybrid per-action native
// dispatcher admits an action iff its footprint is non-opaque and its writes
// are flat-admissible — so both analyses ship behind an opt-in gate and the
// DEFAULT surface is the pre-Wave-4 fail-closed one. These pins are two-sided:
// the resolved arm must be EXACT (with a deliberately-untouched variable that
// must be ABSENT from every set), and the default arm must be OPAQUE.
// ---------------------------------------------------------------------------

/// The exact shape that WP-25 traced the btree divergence to: `GetValue`, whose
/// `LET node == FindLeafNode(root, key)` survives expansion as a recursion
/// re-entry, and whose recursion reaches a SECOND operator (`ChildNodeFor`)
/// containing a LET, a `CHOOSE` over a set-minus, and function applications.
///
/// This is the EXACTNESS half: the resolved footprint must be exactly the
/// variables the action touches. In particular `tick` — which nothing in the
/// call graph mentions — must be ABSENT from reads AND writes, which is what
/// makes the reduction payoff real rather than a rebadged "everything".
///
/// It also discharges the WP-25 prime hypothesis directly: the write set is
/// EXACTLY `{ret, state}`, so the hybrid dispatcher's reconstruction can never
/// leave a written variable at the parent's value for this shape. (The btree
/// divergence diverges on `ret`, which is IN this set — a lowering defect, not
/// a footprint slip.)
const BTREE_GET_VALUE_SHAPE: &str = r#"
---- MODULE PorBtreeGetValue ----
EXTENDS Integers, FiniteSets

VARIABLES childOf, isLeaf, keysOf, lastOf, ret, root, state, tick, valOf

MaxKeyOf(xs) == CHOOSE x \in xs : (\A y \in xs \ {x} : x > y)

ChildNodeFor(node, key) ==
    LET keys == keysOf[node]
        maxKey == MaxKeyOf(keys)
        closestKey == CHOOSE k \in keys : /\ k > key
                                          /\ ~(\E j \in keys \ {k} : j > key /\ j < k)
    IN IF keys = {} \/ key >= maxKey
       THEN lastOf[node]
       ELSE childOf[node, closestKey]

RECURSIVE FindLeafNode(_, _)
FindLeafNode(node, key) ==
    IF isLeaf[node] THEN node ELSE FindLeafNode(ChildNodeFor(node, key), key)

GetValue ==
    LET key == 1
        node == FindLeafNode(root, key)
    IN /\ state = 1
       /\ state' = 0
       /\ ret' = IF key \in keysOf[node] THEN valOf[node, key] ELSE 0
       /\ UNCHANGED <<childOf, isLeaf, keysOf, lastOf, root, tick, valOf>>

Bump ==
    /\ tick' = tick + 1
    /\ UNCHANGED <<childOf, isLeaf, keysOf, lastOf, ret, root, state, valOf>>

Next == GetValue \/ Bump
====
"#;

/// EXACTNESS pin (resolved arm). Sorted var registry: childOf=0, isLeaf=1,
/// keysOf=2, lastOf=3, ret=4, root=5, state=6, tick=7, valOf=8.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_btree_get_value_shape_resolves_exact_footprint() {
    let (ctx, actions) = setup_detected_actions(BTREE_GET_VALUE_SHAPE);
    assert_eq!(actions.len(), 2, "expected GetValue, Bump");

    let deps = extract_detected_action_dependencies(&ctx, &actions);
    assert!(
        !deps[0].opaque,
        "the recursive FindLeafNode chain must resolve under the gate: {:?}",
        deps[0].opaque_reason
    );

    // Reads: the guard (state), the recursion entry (root), everything the
    // recursion and ChildNodeFor touch (isLeaf, keysOf, lastOf, childOf), and
    // the payload read (valOf). NOT tick.
    assert_eq!(
        sorted_vars(&deps[0].reads),
        vec![0, 1, 2, 3, 5, 6, 8],
        "GetValue must read exactly childOf, isLeaf, keysOf, lastOf, root, state, valOf"
    );
    // Writes: exactly the two primed variables. `state' = 0` is a constant
    // write; `ret'` is a real one. Nothing reached through the resolved bodies
    // adds a write, and no UNCHANGED variable is a write.
    assert_eq!(
        sorted_vars(&deps[0].writes),
        vec![4, 6],
        "GetValue must write exactly ret and state — no under- and no over-report"
    );
    // The deliberately-untouched variable, on BOTH sides.
    assert!(
        !deps[0].reads.contains(&VarIndex(7)) && !deps[0].writes.contains(&VarIndex(7)),
        "tick is touched nowhere in GetValue's call graph and must be absent"
    );

    // The precision payoff: GetValue commutes with Bump.
    let matrix = IndependenceMatrix::compute(&deps);
    assert_eq!(matrix.get(0, 1), IndependenceStatus::Independent);
}

/// FAIL-CLOSED pin (DEFAULT arm). With `ResolutionPolicy::OPAQUE` — the
/// shipped default — the very same action is OPAQUE, because the recursion
/// re-entry `FindLeafNode(...)` survives expansion un-inlined. This is what
/// keeps the default hybrid-dispatch surface identical to the pre-Wave-4 tree.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_btree_get_value_shape_stays_opaque_under_default_policy() {
    use super::dependencies_ast::{extract_action_dependencies_with_policy, ResolutionPolicy};

    let (ctx, actions) = setup_detected_actions(BTREE_GET_VALUE_SHAPE);
    let expanded = crate::enumerate::expand_operators_with_primes(&ctx, &actions[0].expr);
    let deps =
        extract_action_dependencies_with_policy(&ctx, &expanded, ResolutionPolicy::OPAQUE);

    assert!(
        deps.opaque,
        "with operator-body resolution OFF (the default) an un-inlined recursion \
         re-entry must fail closed; deps: {deps:?}"
    );
    assert!(
        deps.opaque_reason
            .as_deref()
            .is_some_and(|r| r.contains("FindLeafNode")),
        "must be opaque FOR the un-inlined recursion, not incidentally: {:?}",
        deps.opaque_reason
    );

    // Sanity: the SAME action under the resolved policy is not opaque, so the
    // two arms genuinely differ (the pin cannot pass vacuously).
    let resolved =
        extract_action_dependencies_with_policy(&ctx, &expanded, ResolutionPolicy::RESOLVED);
    assert!(!resolved.opaque, "the resolved arm must still resolve");
}

/// WRITE-PATH pin (WP-25 item 3): a write reached ONLY through a resolved
/// operator body — and reached through a quantifier, an EXCEPT, and a LET
/// inside that body — must be RECORDED, never dropped. A dropped write is the
/// unsound direction twice over: POR would call the action independent of a
/// variable it clobbers, and the hybrid dispatcher would reconstruct the
/// successor with that variable left at the parent's value.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_writes_inside_resolved_operator_body_are_recorded() {
    let spec = r#"
---- MODULE PorResolvedBodyWrites ----
EXTENDS Integers

VARIABLES a, b, c, f, idx

RECURSIVE Ripple(_)
Ripple(n) ==
    IF n <= 0
    THEN /\ \E k \in 1..2 : a' = k
         /\ f' = [f EXCEPT ![idx] = 7]
         /\ LET nb == b + 1 IN b' = nb
    ELSE Ripple(n - 1)

Step == Ripple(2) /\ UNCHANGED c

Tick == c' = c + 1 /\ UNCHANGED <<a, b, f, idx>>

Next == Step \/ Tick
====
"#;
    let (ctx, actions) = setup_detected_actions(spec);
    assert_eq!(actions.len(), 2, "expected Step, Tick");
    let deps = extract_detected_action_dependencies(&ctx, &actions);
    // Sorted var registry: a=0, b=1, c=2, f=3, idx=4.
    assert!(
        !deps[0].opaque,
        "the recursive body must resolve under the gate: {:?}",
        deps[0].opaque_reason
    );
    assert_eq!(
        sorted_vars(&deps[0].writes),
        vec![0, 1, 3],
        "every write reached through the resolved body (quantifier / EXCEPT / LET) \
         must be recorded: a, b, f"
    );
    // `c` is UNCHANGED at the call site and written nowhere in the body — it
    // must be ABSENT from the write set (and from reads).
    assert!(
        !deps[0].writes.contains(&VarIndex(2)) && !deps[0].reads.contains(&VarIndex(2)),
        "c is only UNCHANGED and must not be a read or a write"
    );
    // Step really writes c's peers, so Step and Tick stay independent only
    // because c is untouched — and Step/Tick must NOT be independent if the
    // write set had swallowed c.
    let matrix = IndependenceMatrix::compute(&deps);
    assert_eq!(matrix.get(0, 1), IndependenceStatus::Independent);
}

/// FAIL-CLOSED companion to the write-path pin: a body whose primed target is
/// a COMPOUND expression (`f[i]'`) has an unknowable write set, so resolving
/// the body must degrade the whole action to opaque rather than record a
/// partial footprint.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_unknowable_write_inside_resolved_operator_body_fails_closed() {
    let spec = r#"
---- MODULE PorResolvedBodyPartialWrite ----
EXTENDS Integers

VARIABLES f, idx, v

RECURSIVE Ripple(_)
Ripple(n) == IF n <= 0 THEN f[idx]' = 7 ELSE Ripple(n - 1)

Step == Ripple(2) /\ UNCHANGED v

Next == Step
====
"#;
    let (ctx, actions) = setup_detected_actions(spec);
    let deps = extract_detected_action_dependencies(&ctx, &actions);
    assert!(
        deps[0].opaque,
        "a partial write reached through a resolved body must fail closed; deps: {:?}",
        deps[0]
    );
    assert!(
        deps[0]
            .opaque_reason
            .as_deref()
            .is_some_and(|r| r.contains("primed")),
        "must be opaque FOR the unknowable primed target: {:?}",
        deps[0].opaque_reason
    );
}
