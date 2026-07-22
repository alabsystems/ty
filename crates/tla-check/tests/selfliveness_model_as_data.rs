// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Roadmap step 1 — model-as-DATA discharge of TY's self-liveness obligation.
//!
//! These tests discharge a *structured* `TemporalObligation` (built in code,
//! rendered to TLA+, checked by `check_module`) and confirm it reproduces the
//! engine-selection liveness bug WITHOUT READING the `.tla` file at runtime.
//!
//! Provenance note (no overclaim): the obligation's CONTENT is a hand-transcription
//! of the same oracle (`examples/selfliveness/SelfLivenessJIT.tla`) — it is NOT an
//! independent derivation of TY's behaviour. The on-disk oracle is used here only
//! as the parity REFERENCE for the rendering path, not as independent ground truth.
//!
//! See `crates/tla-check/src/selfliveness/mod.rs` and
//! `docs/design/trust-verification-atoms-2026-06-17.md` §7 step 1.

mod common;

use std::path::Path;

use common::parse_module;
use tla_check::selfliveness::{discharge_tla, TemporalObligation, Verdict, Wiring};
use tla_check::{Config, ConstantValue};

// ===========================================================================
// 1. The bug is detected FROM DATA: bug wiring -> Refuted, lasso ends on the
//    interpreter drain span (native never engaged).
// ===========================================================================

#[cfg_attr(test, ntest::timeout(60000))]
#[test]
fn bug_wiring_refutes_self_liveness_from_structured_atom() {
    let obligation = TemporalObligation::self_liveness_hotness();
    let verdict = obligation.discharge(Wiring::bug());

    match verdict {
        Verdict::Refuted { counterexample } => {
            let property = &counterexample.property;
            let rust_spans = counterexample.rust_spans();
            let actions = counterexample.actions();

            // A P_* liveness property is the one that fails when the work arm is dark.
            assert!(
                property.starts_with("P_"),
                "expected a P_* property to be refuted, got {property:?}"
            );
            // The counterexample must traverse the Rust startup-interp span and
            // TERMINATE on the interpreter drain span — the run finished on the
            // interpreter, never engaging native. That IS the bug.
            assert!(
                rust_spans
                    .iter()
                    .any(|s| s == "run_bfs_notrace.rs:781:startup_interp"),
                "lasso should start at the startup-interp span; got {rust_spans:?}"
            );
            assert!(
                rust_spans.iter().any(|s| s == "run_bfs_loop:drain_interp"),
                "lasso should reach the interpreter-drain span (native never \
                 engaged); got {rust_spans:?}"
            );
            // It must NOT have engaged native anywhere along the counterexample.
            assert!(
                !rust_spans
                    .iter()
                    .any(|s| s == "run_compiled_bfs_loop:drain_native"
                        || s == "run_bfs_notrace.rs:861:hot_swap_to_compiled"),
                "bug lasso must never engage native; got {rust_spans:?}"
            );
            assert!(
                !actions.is_empty(),
                "expected named lasso actions in the counterexample"
            );
            // Visible with `--nocapture`: the structured atom's Rust-span lasso.
            eprintln!("[self-liveness DATA discharge] refuted property: {property}");
            eprintln!("  lasso actions:      {actions:?}");
            eprintln!("  rust spans (lasso): {rust_spans:#?}");
        }
        other => panic!("expected Refuted under bug wiring, got {other:?}"),
    }
}

// ===========================================================================
// 2. The fix is confirmed FROM DATA: fixed wiring -> Verified (exhaustive pass).
// ===========================================================================

#[cfg_attr(test, ntest::timeout(60000))]
#[test]
fn fixed_wiring_verifies_self_liveness_from_structured_atom() {
    let obligation = TemporalObligation::self_liveness_hotness();
    let verdict = obligation.discharge(Wiring::fixed());

    assert!(
        verdict.is_verified(),
        "expected Verified under fixed wiring, got {verdict:?}"
    );
}

// ===========================================================================
// 3. Faithfulness PARITY (strong): the rendered-from-data module must produce
//    the SAME verdict AND the SAME counterexample lasso as the checked-in
//    hand-written oracle, under both wirings. This is `same_outcome`, not merely
//    same verdict CLASS — so two models that both Refute for DIFFERENT reasons
//    would NOT pass. It establishes rendering-path faithfulness to the oracle; it
//    does NOT establish faithfulness to TY's live Rust (that is the still-open
//    MIR-extraction parity test, roadmap step 4).
// ===========================================================================

/// Discharge the on-disk oracle `.tla` with an explicit boolean wiring through
/// the SAME engine (`discharge_tla`) the structured atom uses, so the parity
/// comparison is apples-to-apples (same resolution, same fairness, same checker).
fn discharge_oracle_file(work_arm_wired: bool, hot_swap_wired: bool) -> Verdict {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/selfliveness/SelfLivenessJIT.tla");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read oracle {}: {e}", path.display()));

    let mut constants = std::collections::HashMap::new();
    constants.insert(
        "WorkArmWired".to_string(),
        ConstantValue::Value(if work_arm_wired { "TRUE" } else { "FALSE" }.to_string()),
    );
    constants.insert(
        "HotSwapWired".to_string(),
        ConstantValue::Value(if hot_swap_wired { "TRUE" } else { "FALSE" }.to_string()),
    );
    let config = Config {
        specification: Some("FairSpec".to_string()),
        properties: vec![
            "P_hotness".to_string(),
            "P_artifact_handoff".to_string(),
            "P_reaches_native".to_string(),
        ],
        constants,
        check_deadlock: false,
        ..Default::default()
    };
    discharge_tla(&src, &config)
}

#[cfg_attr(test, ntest::timeout(120000))]
#[test]
fn rendered_atom_matches_oracle_verdict_under_both_wirings() {
    let obligation = TemporalObligation::self_liveness_hotness();

    for (wiring, work_arm, label) in [
        (Wiring::bug(), false, "bug"),
        (Wiring::fixed(), true, "fixed"),
    ] {
        // Discharge the structured atom (render -> resolve -> check).
        let rendered_verdict = obligation.discharge(wiring);
        // Discharge the on-disk oracle with the same wiring, same engine.
        let oracle_verdict = discharge_oracle_file(work_arm, true);

        assert!(
            rendered_verdict.same_outcome(&oracle_verdict),
            "[{label}] rendered-from-data verdict {rendered_verdict:?} disagrees \
             with on-disk oracle verdict {oracle_verdict:?} (same property AND lasso required)"
        );
    }
}

// ===========================================================================
// 4. The renderer is well-formed: parses cleanly, the header block comment is
//    CLOSED (no unterminated `(*`), and the structural landmarks are present.
// ===========================================================================

#[test]
fn rendered_tla_is_well_formed_and_self_describing() {
    let obligation = TemporalObligation::self_liveness_hotness();
    let tla = obligation.render_to_tla();

    // Parses without lowering failure.
    let _module = parse_module(&tla);

    // The header block comment is balanced: equal numbers of `(*` and `*)`.
    let opens = tla.matches("(*").count();
    let closes = tla.matches("*)").count();
    assert_eq!(
        opens, closes,
        "rendered module has unbalanced block comments ({opens} `(*` vs {closes} `*)`):\n{tla}"
    );

    // Structural landmarks.
    for needle in [
        "---- MODULE SelfLivenessJIT ----",
        "CONSTANTS WorkArmWired, HotSwapWired",
        "FairSpec == Spec /\\ Fairness",
        "P_hotness ==",
        // Every action's abstracted Rust span appears in the header map.
        "run_bfs_notrace.rs:781:startup_interp",
        "run_helpers.rs:6765:work_arm_fires",
        "run_bfs_notrace.rs:861:hot_swap_to_compiled",
    ] {
        assert!(
            tla.contains(needle),
            "rendered module missing {needle:?}:\n{tla}"
        );
    }
}
