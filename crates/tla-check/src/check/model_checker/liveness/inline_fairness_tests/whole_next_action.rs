// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for the whole-Next `ActionPred(Next)` fast-path detection
//! (#liveness-whole-next-action).
//!
//! `<<Next>>_vars` decomposes into `ActionPred(Next) /\ StateChanged(vars)`.
//! Every BFS successor edge is produced by Next enumeration, so
//! `ActionPred(Next)` is TRUE on every real successor and the inline recorder
//! sets it directly instead of re-enumerating Next per transition. Detection is
//! gated (fail closed) SOLELY on the fairness action being the config's whole
//! Next (a whole-Next ENABLED leaf was recorded for it) — this TRUE-on-real-edge
//! claim is pure behavior-graph-edge provenance and needs no static proof.
//! (#liveness-whole-next-action-reuse decoupled this from the earlier
//! `action_pins_all_vars` requirement, which is needed only for the separate
//! FALSE-population direction; compound-state whole-Next specs the static prover
//! cannot pin — e.g. YoYoAllGraphs — now engage the fast path too.)
//! A sub-action `WF_vars(A)` where `A` is a PIECE of Next does NOT qualify.

use super::*;

/// Spec whose Next is a disjunction of two fully-pinning sub-actions. We apply
/// weak fairness on BOTH the whole Next AND the sub-action `Inc`, so the same
/// run exercises the positive (whole-Next) and the fail-closed (sub-action)
/// paths side by side.
const SPLIT_NEXT_SPEC: &str = r#"
---- MODULE SplitNext ----
EXTENDS Integers
VARIABLE x, y
Init == x = 0 /\ y = 0
Inc == x' = x + 1 /\ y' = y
Dec == x > 0 /\ x' = x - 1 /\ y' = y
Next == Inc \/ Dec
Prop == []<>(x = 0)
====
"#;

/// The `ActionPred(Next)` leaf of `WF_vars(Next)` (whole Next) is detected as a
/// whole-Next action tag; the `ActionPred(Inc)` leaf of the sub-action
/// `WF_vars(Inc)` is NOT — even though `Inc` also pins all vars — it is not the
/// config's whole Next. The detection must EXACTLY track the whole-Next ENABLED
/// pairing (the fail-closed coupling), independent of pinning.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn whole_next_action_detected_only_for_whole_next() {
    let module = parse_module(SPLIT_NEXT_SPEC);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec!["Prop".to_string()],
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_store_states(true);
    checker.set_fairness(vec![
        FairnessConstraint::Weak {
            vars: "<<x, y>>".to_string(),
            action: "Next".to_string(),
            action_node: None,
        },
        FairnessConstraint::Weak {
            vars: "<<x, y>>".to_string(),
            action: "Inc".to_string(),
            action_node: None,
        },
    ]);

    checker.prepare_inline_fairness_cache();

    let groups = &checker.liveness_cache.enabled_action_groups;
    assert_eq!(
        groups.len(),
        2,
        "WF_vars(Next) + WF_vars(Inc) should produce two enabled-action groups"
    );

    let mut saw_whole_next = false;
    let mut saw_sub_action = false;
    for group in groups {
        let ap = group
            .action_pred_tag
            .expect("each WF group has a unique ActionPred leaf");
        let is_whole_next_enabled = crate::liveness::whole_next_enabled_tag(group.enabled_tag);
        // The action-tag fast path is taken IFF the paired ENABLED is the whole
        // Next relation — the fail-closed coupling.
        assert_eq!(
            crate::liveness::whole_next_action_tag(ap),
            is_whole_next_enabled,
            "whole-Next ActionPred detection must exactly track the whole-Next \
             ENABLED pairing (enabled_tag={}, action_pred_tag={})",
            group.enabled_tag,
            ap,
        );
        if is_whole_next_enabled {
            saw_whole_next = true;
        } else {
            saw_sub_action = true;
        }
    }
    assert!(
        saw_whole_next,
        "WF_vars(Next) should be recognized as a whole-Next action"
    );
    assert!(
        saw_sub_action,
        "WF_vars(Inc) (a PIECE of Next) must NOT be a whole-Next action (fail closed)"
    );
}

/// A single-variable spec whose whole-Next fairness action pins its only
/// variable: `WF_x(Next)` must register the `ActionPred(Next)` leaf as a
/// whole-Next action tag (the positive detection, isolated from the split-Next
/// case above).
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn whole_next_action_detected_for_single_var_next() {
    let module = parse_module(INLINE_FAIRNESS_SPEC);
    let config = inline_fairness_config();

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_store_states(true);
    apply_weak_fairness(&mut checker);

    checker.prepare_inline_fairness_cache();

    let action_tag = find_tag(
        &checker.liveness_cache.fairness_action_checks,
        |expr| match expr {
            LiveExpr::ActionPred { tag, .. } => Some(*tag),
            _ => None,
        },
    );
    assert!(
        crate::liveness::whole_next_action_tag(action_tag),
        "ActionPred(Next) of WF_x(Next) (pins x) should be a whole-Next action tag"
    );
}

/// Spec whose whole Next does NOT statically pin every variable: the `Wrap`
/// disjunct leaves `y'` free, so `action_pins_all_vars(Next)` is FALSE (an `Or`
/// branch that fails to cover every variable fails the proof). This is the
/// miniature of the real compound-state case (YoYoAllGraphs, whose graph `Next`
/// the static prover cannot pin).
const NONPINNING_WHOLE_NEXT_SPEC: &str = r#"
---- MODULE NonPinNext ----
EXTENDS Integers
VARIABLE x, y
Init == x = 0 /\ y = 0
Bump == x' = x + 1 /\ y' = y
Wrap == x = 3 /\ x' = 0
Next == Bump \/ Wrap
Prop == []<>(x = 0)
====
"#;

/// #liveness-whole-next-action-reuse (positive, the decoupling): the
/// `ActionPred(Next)` leaf of `WF_vars(Next)` is a whole-Next action tag PURELY
/// because Next is the config's whole next-state relation — even when the
/// `action_pins_all_vars` proof FAILS (here `Next`'s `Wrap` disjunct leaves `y'`
/// free). Every real BFS successor edge is produced by Next enumeration, so
/// `Next(s, t)` holds by construction; the fast path's TRUE-on-real-edge claim
/// does not depend on pinning. Before the decoupling this leaf fell back to a
/// per-transition Next re-enumeration.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn whole_next_action_detected_without_pinning_proof() {
    use std::sync::Arc;

    let module = parse_module(NONPINNING_WHOLE_NEXT_SPEC);
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        properties: vec!["Prop".to_string()],
        ..Default::default()
    };

    let mut checker = ModelChecker::new(&module, &config);
    checker.set_store_states(true);
    checker.set_fairness(vec![FairnessConstraint::Weak {
        vars: "<<x, y>>".to_string(),
        action: "Next".to_string(),
        action_node: None,
    }]);

    checker.prepare_inline_fairness_cache();

    let group = checker
        .liveness_cache
        .enabled_action_groups
        .iter()
        .find(|g| crate::liveness::whole_next_enabled_tag(g.enabled_tag))
        .expect("WF_vars(Next) must record a whole-Next ENABLED group");
    let ap = group
        .action_pred_tag
        .expect("the whole-Next WF group has a unique ActionPred leaf");

    // Recover the resolved Next action from the paired ENABLED leaf and confirm
    // it genuinely fails the static pinning proof — otherwise this test would
    // silently degenerate into the pins-all-vars case and not exercise the
    // decoupling.
    let action = checker
        .liveness_cache
        .fairness_state_checks
        .iter()
        .find_map(|leaf| match leaf {
            LiveExpr::Enabled { action, tag, .. } if *tag == group.enabled_tag => {
                Some(Arc::clone(action))
            }
            _ => None,
        })
        .expect("whole-Next ENABLED leaf must be present in the state checks");
    let var_names: Vec<Arc<str>> = vec![Arc::from("x"), Arc::from("y")];
    assert!(
        !crate::liveness::action_pins_all_vars(&action, &var_names, None),
        "the NonPinNext spec's Next must NOT statically pin all vars — otherwise \
         this test does not exercise the pinning-decoupled path"
    );

    assert!(
        crate::liveness::whole_next_action_tag(ap),
        "ActionPred(Next) must be a whole-Next action tag from the whole-Next \
         property ALONE, even though action_pins_all_vars(Next) is false"
    );
}
