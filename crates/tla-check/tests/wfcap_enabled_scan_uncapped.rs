// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Regression test: the capped explored-successor predicate scan must never be
//! the FINAL ENABLED decision procedure for a fairness action.
//!
//! `cached_successor_satisfies_action` caps its predicate evaluations (at 2)
//! on the assumption that every `false` result is followed by an authoritative
//! enumeration. That assumption fails for non-enumerable ("unpinnable")
//! actions such as `A == x + x' = 5 + x` (semantically `x' = 5`, but the
//! primed variable is buried in arithmetic so the enumeration cannot pin it):
//! the under-specification rescue scan IS the final answer there. With the cap
//! applied at that call site, a state whose witnessing successor sits behind
//! two non-witnessing explored successors is wrongly deemed to have `A`
//! DISABLED, the `WF_<<x>>(A)` obligation is dropped, and a HOLDING liveness
//! property is reported as violated.
//!
//! From x=0 the explored successors are 1, 2 ((x+1)%3, (x+2)%3) and then 5 —
//! the `x'=5` witness is the THIRD scanned successor, exactly past the cap.
//! WF_<<x>>(A) forces `x'=5` to eventually fire, so `<>(x = 5)` HOLDS.

use tla_check::Config;
use tla_check::{resolve_spec_from_config, CheckResult};
use tla_core::{lower, parse_to_syntax_tree, FileId};

mod common;

/// Helper: parse, resolve, and check a spec with a PROPERTY config
/// (equivalent to cfg `SPECIFICATION Spec` / `PROPERTY Prop`).
fn check_liveness_spec(src: &str, property: &str) -> CheckResult {
    let _guard = common::EnvVarGuard::remove("TY_SKIP_LIVENESS");

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

/// `A == x + x' = 5 + x` (i.e. `x' = 5`) is genuinely enabled at every state,
/// but only its THIRD explored successor witnesses it at x=0. WF_<<x>>(A)
/// therefore forces `x = 5` eventually: `<>(x = 5)` HOLDS. A capped final
/// scan wrongly reports A disabled and fabricates a liveness violation.
#[cfg_attr(test, ntest::timeout(60000))]
#[test]
fn wf_on_unpinnable_action_witness_past_scan_cap_holds() {
    let src = r#"
---- MODULE WfCap ----
EXTENDS Integers

VARIABLE x

Init == x = 0

Next == x' = (x + 1) % 3 \/ x' = (x + 2) % 3 \/ x' = 5 \/ (x = 5 /\ x' = 0)

A == x + x' = 5 + x

Prop == <>(x = 5)

Spec == Init /\ [][Next]_<<x>> /\ WF_<<x>>(A)

====
"#;
    let result = check_liveness_spec(src, "Prop");
    match &result {
        CheckResult::Success(_) => {
            // Correct: WF_<<x>>(A) with A always enabled (x'=5 is always a
            // successor) forces the x'=5 transition eventually; <>(x=5) holds.
        }
        CheckResult::LivenessViolation { .. } => {
            panic!(
                "FALSE liveness counterexample: <>(x = 5) HOLDS under WF_<<x>>(A) \
                 (A == x + x' = 5 + x is always enabled), but the checker reported a \
                 violation — the capped explored-successor scan was used as the final \
                 ENABLED decision for a non-enumerable action"
            );
        }
        other => panic!("expected Success for <>(x = 5) with WF_<<x>>(A), got: {other:?}"),
    }
}
