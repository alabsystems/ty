// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Regression tests for #liveness-ea-gate (N5): the counterexample witness gate
//! must re-verify a PEM's EA (`<>[]c`) conjuncts with the AUTHORITATIVE
//! interpreter, and must RUN even for EA-only PEMs.
//!
//! Historically a PEM's EA conjuncts were enforced ONLY by the Tarjan/witness
//! edge filter reading the per-node `state_check_mask` / `action_check_masks`
//! bitmasks — the very bitmasks the gate declares untrustworthy for ENABLED —
//! and the gate was SKIPPED entirely for EA-only PEMs. A bitmask false-positive
//! on an EA conjunct therefore fabricated an unsound liveness counterexample.
//!
//! These tests inject a POISONED inline bitmask directly (bypassing the mask
//! populator) so the EA edge filter keeps a cycle the authoritative interpreter
//! rejects, isolating the gate behaviour from the rest of the pipeline.

use super::helpers::state_pred_x_eq;
use super::*;
use crate::liveness::test_helpers::{empty_successors, make_checker};
use crate::liveness::LiveExpr;
use crate::Value;

/// EA-only PEM (`<>[](x=0)`) over the 2-cycle x=0 <-> x=1. `<>[](x=0)` requires
/// x=0 at EVERY cycle node, so it is genuinely UNSATISFIABLE (the cycle visits
/// x=1) — the correct verdict is Satisfied (property holds).
///
/// The injected bitmask has a FALSE POSITIVE on x=1 (the `x=0` check bit set
/// there too), so the EA edge filter keeps both edges and Tarjan finds the SCC.
/// A gate that trusts the EA conjunct from the bitmask (or is skipped for this
/// EA-only PEM) reports a false Violation; the authoritative EA re-verification
/// evaluates `x=0` at the x=1 node, finds it false, and refutes the witness.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn ea_only_poisoned_bitmask_does_not_fabricate_violation() {
    let mut checker = make_checker(LiveExpr::always(LiveExpr::Bool(true)));
    let mut get_successors = empty_successors;

    let s0 = State::from_pairs([("x", Value::int(0))]);
    let s1 = State::from_pairs([("x", Value::int(1))]);

    let init_nodes = checker
        .add_initial_state(&s0, &mut get_successors, None)
        .expect("add_initial_state");
    let from_s0 = checker
        .add_successors(
            init_nodes[0],
            std::slice::from_ref(&s1),
            &mut get_successors,
            None,
        )
        .expect("s0->s1");
    let _ = checker
        .add_successors(
            from_s0[0],
            std::slice::from_ref(&s0),
            &mut get_successors,
            None,
        )
        .expect("s1->s0");

    let tag: u32 = 1;
    let plan = GroupedLivenessPlan {
        tf: LiveExpr::Bool(true),
        check_state: vec![state_pred_x_eq(0, tag)], // idx 0: EA state <>[](x=0)
        check_action: Vec::new(),
        pems: vec![PemPlan {
            ea_state_idx: vec![0],
            ea_action_idx: Vec::new(),
            ae_state_idx: Vec::new(),
            ae_action_idx: Vec::new(),
        }],
    };

    let mut state_bm: FxHashMap<crate::state::Fingerprint, u64> = FxHashMap::default();
    state_bm.insert(s0.fingerprint(), 1u64 << tag); // TRUE: x=0 holds at x=0
    state_bm.insert(s1.fingerprint(), 1u64 << tag); // FALSE POSITIVE: x=0 at x=1
    let action_bm: FxHashMap<(crate::state::Fingerprint, crate::state::Fingerprint), u64> =
        FxHashMap::default();
    let inline = InlineCheckResults {
        max_tag: tag,
        state_bitmasks: &state_bm,
        action_bitmasks: &action_bm,
    };

    let result = checker.check_liveness_grouped_with_inline_cache(&plan, 0, Some(inline), None);
    assert!(
        matches!(result, LivenessResult::Satisfied),
        "<>[](x=0) is unsatisfiable in the x=0<->x=1 SCC; the poisoned EA bitmask \
         must NOT fabricate a violation — the gate must re-verify the EA conjunct \
         authoritatively. Got: {result:?}"
    );
}

/// Positive control / no over-refutation: a GENUINE EA violation is still
/// reported after the fix. Self-loop x=0 -> x=0 with EA `<>[](x=0)`: `x=0` holds
/// at every cycle node, so the property is genuinely violated (there is a fair
/// suffix where it always holds) and the checker must report Violated.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn ea_only_genuine_violation_still_reported() {
    let mut checker = make_checker(LiveExpr::always(LiveExpr::Bool(true)));
    let mut get_successors = empty_successors;

    let s0 = State::from_pairs([("x", Value::int(0))]);
    let init_nodes = checker
        .add_initial_state(&s0, &mut get_successors, None)
        .expect("add_initial_state");
    // Self-loop x=0 -> x=0.
    let _ = checker
        .add_successors(
            init_nodes[0],
            std::slice::from_ref(&s0),
            &mut get_successors,
            None,
        )
        .expect("s0->s0");

    let tag: u32 = 1;
    let plan = GroupedLivenessPlan {
        tf: LiveExpr::Bool(true),
        check_state: vec![state_pred_x_eq(0, tag)], // idx 0: EA state <>[](x=0)
        check_action: Vec::new(),
        pems: vec![PemPlan {
            ea_state_idx: vec![0],
            ea_action_idx: Vec::new(),
            ae_state_idx: Vec::new(),
            ae_action_idx: Vec::new(),
        }],
    };

    let mut state_bm: FxHashMap<crate::state::Fingerprint, u64> = FxHashMap::default();
    state_bm.insert(s0.fingerprint(), 1u64 << tag); // ACCURATE: x=0 holds at x=0
    let action_bm: FxHashMap<(crate::state::Fingerprint, crate::state::Fingerprint), u64> =
        FxHashMap::default();
    let inline = InlineCheckResults {
        max_tag: tag,
        state_bitmasks: &state_bm,
        action_bitmasks: &action_bm,
    };

    let result = checker.check_liveness_grouped_with_inline_cache(&plan, 0, Some(inline), None);
    assert!(
        matches!(
            result,
            LivenessResult::Violated { .. } | LivenessResult::ViolatedFingerprints { .. }
        ),
        "<>[](x=0) genuinely holds on the x=0 self-loop; the EA re-verification \
         must CONFIRM (not over-refute) this witness. Got: {result:?}"
    );
}
