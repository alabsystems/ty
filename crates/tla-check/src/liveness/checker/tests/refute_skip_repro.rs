// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Regression test for #liveness-refute-skip (N11): when the AE gate
//! authoritatively refutes the single bitmask-constructed witness cycle,
//! `check_liveness_grouped` used to skip the WHOLE SCC even though a DIFFERENT
//! cycle in the same SCC genuinely satisfies the AE constraints (the milestone
//! bitmasks are fallible in the false-POSITIVE direction). The authoritative
//! fallback cycle builder must find the genuine cycle before skipping the SCC.

use super::helpers::{action_pred_xprime_eq_x_plus_1, state_pred_x_eq};
use super::*;
use crate::liveness::tarjan::Scc;
use crate::liveness::test_helpers::{empty_successors, make_checker, make_checker_with_vars};
use crate::liveness::LiveExpr;
use crate::Value;

/// Build the 3-node SCC:  A(x=0) -> C(x=1) -> A,  C -> B(x=3) -> A.
/// Returns (checker, fp_a, fp_b, fp_c).
fn build_scc_checker() -> (
    LivenessChecker,
    crate::state::Fingerprint,
    crate::state::Fingerprint,
    crate::state::Fingerprint,
) {
    let mut checker = make_checker(LiveExpr::always(LiveExpr::Bool(true)));
    let mut get_successors = empty_successors;

    let sa = State::from_pairs([("x", Value::int(0))]);
    let sc = State::from_pairs([("x", Value::int(1))]);
    let sb = State::from_pairs([("x", Value::int(3))]);

    let init_nodes = checker
        .add_initial_state(&sa, &mut get_successors, None)
        .expect("add_initial_state");
    let na = init_nodes[0];
    // A -> C
    let nc_nodes = checker
        .add_successors(na, std::slice::from_ref(&sc), &mut get_successors, None)
        .expect("A->C");
    let nc = nc_nodes[0];
    // C -> A first (so the BFS return path A..A goes A->C->A and avoids B),
    // then C -> B.
    let _ = checker
        .add_successors(nc, std::slice::from_ref(&sa), &mut get_successors, None)
        .expect("C->A");
    let nb_nodes = checker
        .add_successors(nc, std::slice::from_ref(&sb), &mut get_successors, None)
        .expect("C->B");
    let nb = nb_nodes[0];
    // B -> A
    let _ = checker
        .add_successors(nb, std::slice::from_ref(&sa), &mut get_successors, None)
        .expect("B->A");

    (
        checker,
        sa.fingerprint(),
        sb.fingerprint(),
        sc.fingerprint(),
    )
}

fn plan_ae_x_eq_3(tag: u32) -> GroupedLivenessPlan {
    GroupedLivenessPlan {
        tf: LiveExpr::Bool(true),
        check_state: vec![state_pred_x_eq(3, tag)],
        check_action: Vec::new(),
        pems: vec![PemPlan {
            ae_state_idx: vec![0],
            ea_state_idx: Vec::new(),
            ae_action_idx: Vec::new(),
            ea_action_idx: Vec::new(),
        }],
    }
}

/// CONTROL: with an ACCURATE inline bitmask (bit set exactly on B, where
/// x=3 genuinely holds), the checker reports Violated — proving the SCC
/// contains a genuine AE-satisfying cycle (A -> C -> B -> A).
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn control_accurate_mask_reports_violation() {
    let (mut checker, fp_a, fp_b, fp_c) = build_scc_checker();
    let tag: u32 = 1;
    let plan = plan_ae_x_eq_3(tag);

    let mut state_bm: FxHashMap<crate::state::Fingerprint, u64> = FxHashMap::default();
    state_bm.insert(fp_a, 0);
    state_bm.insert(fp_c, 0);
    state_bm.insert(fp_b, 1u64 << tag); // TRUE positive on B
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
        "accurate mask must find the genuine violation, got: {result:?}"
    );
}

/// REGRESSION (N11): same graph, same genuine violation, but the inline bitmask
/// has a FALSE POSITIVE on A (bit set, authoritative x=3 false there) and a
/// false negative on B. The bitmask witness cycle is built through A, avoids B,
/// and the gate authoritatively refutes it. Before the fix, the loop `continue`d
/// to the next SCC and the genuine cycle A -> C -> B -> A in the SAME SCC was
/// never tried (false HOLD). The authoritative fallback builder must now
/// re-select milestone B by real evaluation and report the genuine violation.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn poisoned_mask_on_wrong_node_misses_genuine_violation_in_same_scc() {
    let (mut checker, fp_a, fp_b, fp_c) = build_scc_checker();
    let tag: u32 = 1;
    let plan = plan_ae_x_eq_3(tag);

    let mut state_bm: FxHashMap<crate::state::Fingerprint, u64> = FxHashMap::default();
    state_bm.insert(fp_a, 1u64 << tag); // FALSE positive on A
    state_bm.insert(fp_c, 0);
    state_bm.insert(fp_b, 0); // false negative on B (genuinely x=3)
    let action_bm: FxHashMap<(crate::state::Fingerprint, crate::state::Fingerprint), u64> =
        FxHashMap::default();
    let inline = InlineCheckResults {
        max_tag: tag,
        state_bitmasks: &state_bm,
        action_bitmasks: &action_bm,
    };

    let result = checker.check_liveness_grouped_with_inline_cache(&plan, 0, Some(inline), None);
    // The CORRECT verdict for this behavior graph is Violated (see control):
    // the authoritative fallback must find the genuine cycle A -> C -> B -> A
    // rather than skipping the SCC after the bitmask candidate was refuted.
    assert!(
        matches!(
            result,
            LivenessResult::Violated { .. } | LivenessResult::ViolatedFingerprints { .. }
        ),
        "genuine violation in the SCC must be found via the authoritative fallback \
         after the gate refuted the bitmask-constructed candidate cycle; got: {result:?}"
    );
}

fn build_owned_two_node_scc() -> (
    LivenessChecker,
    Scc,
    BehaviorGraphNode,
    BehaviorGraphNode,
) {
    let mut checker =
        make_checker_with_vars(LiveExpr::always(LiveExpr::Bool(true)), &["x"]);
    checker.enable_owned_behavior_graph_state_cache();
    let mut get_successors = empty_successors;
    let s0 = State::from_pairs([("x", Value::int(0))]);
    let s1 = State::from_pairs([("x", Value::int(1))]);

    let n0 = checker
        .add_initial_state(&s0, &mut get_successors, None)
        .expect("owned initial state")[0];
    let n1 = checker
        .add_successors(n0, std::slice::from_ref(&s1), &mut get_successors, None)
        .expect("owned 0->1 successor")[0];
    checker
        .add_successors(n1, std::slice::from_ref(&s0), &mut get_successors, None)
        .expect("owned 1->0 successor");
    checker
        .state_successor_fps
        .insert(s0.fingerprint(), std::sync::Arc::new(vec![s1.fingerprint()]));
    checker
        .state_successor_fps
        .insert(s1.fingerprint(), std::sync::Arc::new(vec![s0.fingerprint()]));

    (checker, Scc::new(vec![n0, n1]), n0, n1)
}

fn authoritative_state_plan() -> GroupedLivenessPlan {
    GroupedLivenessPlan {
        tf: LiveExpr::Bool(true),
        check_state: vec![state_pred_x_eq(0, 1)],
        check_action: Vec::new(),
        pems: vec![PemPlan {
            ae_state_idx: vec![0],
            ea_state_idx: Vec::new(),
            ae_action_idx: Vec::new(),
            ea_action_idx: Vec::new(),
        }],
    }
}

fn authoritative_action_plan() -> GroupedLivenessPlan {
    GroupedLivenessPlan {
        tf: LiveExpr::Bool(true),
        check_state: Vec::new(),
        check_action: vec![action_pred_xprime_eq_x_plus_1(1)],
        pems: vec![PemPlan {
            ae_state_idx: Vec::new(),
            ea_state_idx: Vec::new(),
            ae_action_idx: vec![0],
            ea_action_idx: Vec::new(),
        }],
    }
}

fn empty_authoritative_plan() -> GroupedLivenessPlan {
    GroupedLivenessPlan {
        tf: LiveExpr::Bool(true),
        check_state: Vec::new(),
        check_action: Vec::new(),
        pems: vec![PemPlan {
            ae_state_idx: Vec::new(),
            ea_state_idx: Vec::new(),
            ae_action_idx: Vec::new(),
            ea_action_idx: Vec::new(),
        }],
    }
}

#[test]
fn authoritative_fallback_owned_ae_state_missing_source_payload_fails_closed() {
    let (mut checker, scc, n0, _) = build_owned_two_node_scc();
    checker.graph.remove_owned_state_for_test(n0.state_fp);
    let plan = authoritative_state_plan();

    let error = checker
        .build_witness_cycle_in_scc_authoritative(&scc, None, &plan, &plan.pems[0])
        .expect_err("owned AE-state source payload loss must fail closed");

    assert!(error
        .to_string()
        .contains("authoritative AE-state source is missing payload"));
}

#[test]
fn authoritative_fallback_owned_ae_action_missing_source_payload_fails_closed() {
    let (mut checker, scc, n0, _) = build_owned_two_node_scc();
    checker.graph.remove_owned_state_for_test(n0.state_fp);
    let plan = authoritative_action_plan();

    let error = checker
        .build_witness_cycle_in_scc_authoritative(&scc, None, &plan, &plan.pems[0])
        .expect_err("owned AE-action source payload loss must fail closed");

    assert!(error
        .to_string()
        .contains("authoritative AE-action source is missing payload"));
}

#[test]
fn authoritative_fallback_owned_ae_action_missing_destination_payload_fails_closed() {
    let (mut checker, scc, _, n1) = build_owned_two_node_scc();
    checker.graph.remove_owned_state_for_test(n1.state_fp);
    let plan = authoritative_action_plan();

    let error = checker
        .build_witness_cycle_in_scc_authoritative(&scc, None, &plan, &plan.pems[0])
        .expect_err("owned AE-action destination payload loss must fail closed");

    assert!(error
        .to_string()
        .contains("authoritative AE-action destination is missing payload"));
}

#[test]
fn authoritative_gate_owned_missing_cycle_payload_fails_closed() {
    let (mut checker, _scc, n0, n1) = build_owned_two_node_scc();
    checker.graph.remove_owned_state_for_test(n0.state_fp);
    let plan = empty_authoritative_plan();

    let error = checker
        .witness_cycle_satisfies_pem(&[n0, n1], &plan, &plan.pems[0])
        .expect_err("owned authoritative gate payload loss must be an invariant error");

    assert!(error
        .to_string()
        .contains("missing authoritative cycle payload"));
}

#[test]
fn authoritative_gate_rejects_non_edge_cycle_pair() {
    let (mut checker, _scc, n0, _) = build_owned_two_node_scc();
    let plan = empty_authoritative_plan();

    let error = checker
        .witness_cycle_satisfies_pem(&[n0, n0], &plan, &plan.pems[0])
        .expect_err("a non-edge consecutive cycle pair must fail closed");

    assert!(error.to_string().contains("is not a behavior-graph edge"));
}

#[test]
fn authoritative_gate_rejects_missing_cycle_source_node_info() {
    let (mut checker, _scc, n0, n1) = build_owned_two_node_scc();
    let missing_node = BehaviorGraphNode {
        state_fp: n0.state_fp,
        tableau_idx: usize::MAX,
    };
    let plan = empty_authoritative_plan();

    let error = checker
        .witness_cycle_satisfies_pem(&[missing_node, n1], &plan, &plan.pems[0])
        .expect_err("a missing cycle source node must fail closed");

    assert!(error
        .to_string()
        .contains("authoritative cycle source node"));
}

#[test]
fn owned_confirmed_cycle_missing_payload_never_downgrades_to_fingerprints() {
    let (mut checker, _scc, n0, _) = build_owned_two_node_scc();
    checker.graph.remove_owned_state_for_test(n0.state_fp);
    let result = checker.violation_result_for_cycle(&[n0]);

    assert!(
        matches!(
            result,
            LivenessResult::RuntimeFailure { ref reason }
                if reason.contains("counterexample trace") && reason.contains("missing")
        ),
        "owned payload loss must surface as a runtime invariant failure, got {result:?}"
    );
}

#[test]
fn empty_confirmed_cycle_fails_closed_without_panicking() {
    let (checker, _scc, _n0, _n1) = build_owned_two_node_scc();
    let result = checker.violation_result_for_cycle(&[]);

    assert!(
        matches!(
            result,
            LivenessResult::RuntimeFailure { ref reason }
                if reason.contains("empty confirmed witness cycle")
        ),
        "an empty confirmed witness cycle must be rejected, got {result:?}"
    );
}

#[test]
fn authoritative_gate_checks_single_node_self_loop_action_constraints() {
    let (mut checker, _scc, n0, _) = build_owned_two_node_scc();
    checker
        .graph
        .get_node_info_mut(&n0)
        .expect("owned node info")
        .successors
        .push(n0);

    for (ae_action_idx, ea_action_idx) in [(vec![0], Vec::new()), (Vec::new(), vec![0])] {
        let plan = GroupedLivenessPlan {
            tf: LiveExpr::Bool(true),
            check_state: Vec::new(),
            check_action: vec![LiveExpr::Bool(false)],
            pems: vec![PemPlan {
                ae_state_idx: Vec::new(),
                ea_state_idx: Vec::new(),
                ae_action_idx,
                ea_action_idx,
            }],
        };

        let verdict = checker
            .witness_cycle_satisfies_pem(&[n0], &plan, &plan.pems[0])
            .expect("single-node self-loop gate evaluation");
        assert!(matches!(
            verdict,
            super::super::checks::WitnessAeVerdict::Refuted
        ));
    }
}
