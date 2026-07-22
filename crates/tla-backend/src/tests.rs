// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! Truth-table tests pinning the live shims: the `legacy_env_plan` synthesis contract
//! and the canonical `TY_*` flag truth table.

use crate::env_overlay::{legacy_env_plan, LegacyEnvPlan};
use crate::request::{EngineId, EngineRequest, SelectionMode};

// --- byte-identical contract: legacy_env_plan per SelectionMode ---

#[test]
fn legacy_env_plan_auto_sets_both() {
    let plan = legacy_env_plan(&EngineRequest::for_check(SelectionMode::Auto));
    assert_eq!(
        plan,
        LegacyEnvPlan {
            set_trust_cg_bfs: true,
            set_auto_select: true,
        }
    );
}

#[test]
fn legacy_env_plan_forced_native_sets_only_bfs() {
    let plan = legacy_env_plan(&EngineRequest::for_check(SelectionMode::Forced(
        EngineId::TrustCgNative,
    )));
    assert_eq!(
        plan,
        LegacyEnvPlan {
            set_trust_cg_bfs: true,
            set_auto_select: false,
        }
    );
}

#[test]
fn legacy_env_plan_oracle_sets_nothing() {
    let plan = legacy_env_plan(&EngineRequest::for_check(SelectionMode::Oracle));
    assert_eq!(
        plan,
        LegacyEnvPlan {
            set_trust_cg_bfs: false,
            set_auto_select: false,
        }
    );
}

// --- canonical TY_* flag truth table ---

#[test]
fn canonical_env_flag_truth_table() {
    use crate::env_overlay::{env_flag_disabled, env_flag_enabled};
    // enabled: trimmed == "1"
    assert!(env_flag_enabled("1"));
    assert!(env_flag_enabled("  1 "));
    assert!(!env_flag_enabled("0"));
    assert!(!env_flag_enabled("true"));
    assert!(!env_flag_enabled(""));
    // disabled: 0/false/off/no (case-insensitive)
    assert!(env_flag_disabled("0"));
    assert!(env_flag_disabled("false"));
    assert!(env_flag_disabled("OFF"));
    assert!(env_flag_disabled("No"));
    assert!(!env_flag_disabled("1"));
    assert!(!env_flag_disabled("yes"));
}
