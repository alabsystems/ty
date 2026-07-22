// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Soundness tests for the per-state batched ENABLED evaluation context
//! (#liveness-enabled-batch-ctx).
//!
//! The batched context (`prepare_enabled_ctx` + `EnabledEvalRequest::state_prepared`)
//! amortizes the per-leaf current-state snapshot+rebind over all ENABLED leaves
//! of one state. It only engages when a state carries MORE THAN ONE ENABLED
//! leaf (`eval_enabled_leaves_batched`), so every spec here declares at least
//! two `WF_`/`SF_` fairness conjuncts to force the batch path.
//!
//! Both directions are checked:
//!   * fail-closed — a genuinely-violated liveness property is still reported as
//!     a violation (the batch path must not spuriously mark actions enabled and
//!     make WF/SF trivially satisfiable);
//!   * positive — a holding liveness property is still reported as holding (the
//!     batch path must not spuriously mark actions disabled and fabricate an
//!     unfair cycle).
//!
//! The `TY_DISABLE_LIVENESS_ENABLED_BATCH=1` kill switch (verified out-of-process
//! against the corpus) forces the legacy per-leaf preparation; these in-process
//! tests exercise the DEFAULT (batched) path.

use tla_check::Config;
use tla_check::{resolve_spec_from_config, CheckResult};
use tla_core::{lower, parse_to_syntax_tree, FileId};

mod common;

/// Parse, resolve, and check `src` against `property`, using `Spec`'s fairness.
fn check_liveness_spec(src: &str, property: &str) -> CheckResult {
    let _guard = common::EnvVarGuard::remove("TY_SKIP_LIVENESS");
    // Belt-and-suspenders: ensure the batched path is active (default) even if
    // the ambient environment disabled it.
    let _batch_guard = common::EnvVarGuard::remove("TY_DISABLE_LIVENESS_ENABLED_BATCH");

    let tree = parse_to_syntax_tree(src);
    let lower_result = lower(FileId(0), &tree);
    let module = lower_result.module.expect("module lowering should succeed");

    let spec_config = Config {
        specification: Some("Spec".to_string()),
        ..Default::default()
    };
    let resolved =
        resolve_spec_from_config(&spec_config, &tree).expect("spec resolution should succeed");

    let config = Config {
        init: Some(resolved.init.clone()),
        next: Some(resolved.next.clone()),
        properties: vec![property.to_string()],
        ..Default::default()
    };

    let mut checker = tla_check::ModelChecker::new(&module, &config);
    checker.set_deadlock_check(false);
    checker.set_store_states(true);
    checker.set_fairness(resolved.fairness);
    checker.set_stuttering_allowed(resolved.stuttering_allowed);
    checker.check()
}

/// Fail-closed: two WF conjuncts (WF(A), WF(B)) — two ENABLED state leaves per
/// state, so the batch path engages — but the checked property `<>(z=1)` is NOT
/// guaranteed by that fairness (nothing forces C, which is the only action that
/// sets `z`). A fair behavior takes A, then B, then Idle forever with `z=0`, so
/// the property is violated and MUST be reported as such. If the batched ENABLED
/// context wrongly reported A or B as always-enabled/disabled the WF obligations
/// (and hence the violating cycle) would be miscomputed.
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn batch_ctx_preserves_multi_wf_violation() {
    let src = r#"
---- MODULE BatchCtxViolation ----
VARIABLES x, y, z
vars == <<x, y, z>>

Init == x = 0 /\ y = 0 /\ z = 0

A == x = 0 /\ x' = 1 /\ y' = y /\ z' = z
B == y = 0 /\ y' = 1 /\ x' = x /\ z' = z
C == z = 0 /\ z' = 1 /\ x' = x /\ y' = y
Idle == UNCHANGED vars

Next == A \/ B \/ C \/ Idle

Spec == Init /\ [][Next]_vars /\ WF_vars(A) /\ WF_vars(B)

Live == <>(z = 1)
====
"#;
    let result = check_liveness_spec(src, "Live");
    match &result {
        CheckResult::LivenessViolation { .. } => {}
        other => panic!(
            "expected LivenessViolation for <>(z=1) under WF(A)/\\WF(B) only \
             (batch path must not hide the unfair z=0 cycle), got: {other:?}"
        ),
    }
}

/// Positive: adding WF(C) — three WF conjuncts, three ENABLED state leaves per
/// state (batch path engaged) — forces C to fire, so `z` eventually becomes 1
/// and `<>(z=1)` HOLDS. The batched ENABLED context must preserve this hold (it
/// must not spuriously mark C disabled and fabricate an unfair cycle).
#[cfg_attr(test, ntest::timeout(30000))]
#[test]
fn batch_ctx_preserves_multi_wf_hold() {
    let src = r#"
---- MODULE BatchCtxHold ----
VARIABLES x, y, z
vars == <<x, y, z>>

Init == x = 0 /\ y = 0 /\ z = 0

A == x = 0 /\ x' = 1 /\ y' = y /\ z' = z
B == y = 0 /\ y' = 1 /\ x' = x /\ z' = z
C == z = 0 /\ z' = 1 /\ x' = x /\ y' = y
Idle == UNCHANGED vars

Next == A \/ B \/ C \/ Idle

Spec == Init /\ [][Next]_vars /\ WF_vars(A) /\ WF_vars(B) /\ WF_vars(C)

Live == <>(z = 1)
====
"#;
    let result = check_liveness_spec(src, "Live");
    match &result {
        CheckResult::Success(_) => {}
        other => panic!(
            "expected Success for <>(z=1) under WF(A)/\\WF(B)/\\WF(C) \
             (batch path must preserve the WF(C)-forced hold), got: {other:?}"
        ),
    }
}
