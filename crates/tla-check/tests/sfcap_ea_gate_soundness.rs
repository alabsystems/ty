// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Regression test (#liveness-ea-gate / N5): the counterexample witness gate
//! must re-verify a PEM's EA (`<>[]c`) conjuncts with the AUTHORITATIVE
//! interpreter, not trust them straight from the distrusted per-node check
//! bitmasks.
//!
//! A PEM's EA conjuncts were historically enforced ONLY by the Tarjan/witness
//! edge filter reading `state_check_mask` / `action_check_masks` — the very
//! bitmasks the gate's own soundness comment declares untrustworthy for
//! ENABLED — and the gate was SKIPPED entirely for EA-only PEMs. Strong
//! fairness `SF_<<x>>(A)` contributes the EA-state conjunct `<>[]~ENABLED<A>`;
//! a bitmask false-positive on `~ENABLED` (the action's witnessing successor
//! absent from the recorded slice) makes the conjunct trivially satisfiable and
//! fabricates a liveness counterexample for a HOLDING property.
//!
//! `A == x' = 5` is enabled at every state (`x' = 5` is always a `Next`
//! disjunct), so `SF_<<x>>(A)` forces `x' = 5` to fire infinitely often and
//! `<>(x = 5)` HOLDS. The gate now re-verifies `<>[]~ENABLED<A>` authoritatively
//! (ENABLED resolved against the complete post-BFS successor set), refuting the
//! bitmask false-positive, so no false violation is reported.

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

/// `A == x' = 5` is always enabled, so `SF_<<x>>(A)` forces `x = 5` infinitely
/// often and `<>(x = 5)` HOLDS. A gate that trusts the EA conjunct
/// `<>[]~ENABLED<A>` straight from the distrusted bitmasks (or skips the gate
/// for this EA-only PEM) fabricates a violation; the authoritative EA
/// re-verification refutes it.
#[cfg_attr(test, ntest::timeout(60000))]
#[test]
fn sf_cap_ea_conjunct_reverified_property_holds() {
    let src = r#"
---- MODULE SfCap ----
EXTENDS Integers

VARIABLE x

Init == x = 0

Next == x' = (x + 1) % 3 \/ x' = (x + 2) % 3 \/ x' = 5 \/ (x = 5 /\ x' = 0)

A == x' = 5

Prop == <>(x = 5)

Spec == Init /\ [][Next]_<<x>> /\ SF_<<x>>(A)

====
"#;
    let result = check_liveness_spec(src, "Prop");
    match &result {
        CheckResult::Success(_) => {
            // Correct: SF_<<x>>(A) with A always enabled forces x'=5 to fire
            // infinitely often; <>(x=5) holds.
        }
        CheckResult::LivenessViolation { .. } => {
            panic!(
                "FALSE liveness counterexample: <>(x = 5) HOLDS under SF_<<x>>(A) \
                 (A == x' = 5 is always enabled), but the checker reported a violation — \
                 the EA conjunct <>[]~ENABLED<A> was trusted from the distrusted bitmasks \
                 instead of being re-verified by the authoritative interpreter"
            );
        }
        other => panic!("expected Success for <>(x = 5) with SF_<<x>>(A), got: {other:?}"),
    }
}
