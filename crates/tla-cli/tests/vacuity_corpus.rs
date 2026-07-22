// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Vacuity-gate confirmation corpus (TRUST_VACUITY_GATE §3.1).
//!
//! Each "red" fixture is a deliberately-vacuous spec that MUST flag with its
//! exact exit code / verdict; each control is the corresponding genuine spec
//! that MUST stay silent. We assert exact exit codes (0 / 3) and the verdict /
//! warning text via the real `ty` binary.
//!
//! Exit-code contract:
//! - `0` = Success (and non-fatal V2/V3 WARNINGs may be present on stderr)
//! - `3` = VACUOUS (distinct from `1` = FAILED / property violation)

mod common;

use common::{run_tla_parsed, write_spec_and_config, TempDir};

/// Run `ty check <spec> --config <cfg> [extra...]`, returning (code, stdout, stderr).
fn check(
    dir: &TempDir,
    module: &str,
    spec: &str,
    cfg: &str,
    extra: &[&str],
) -> (i32, String, String) {
    let (spec_path, cfg_path) = write_spec_and_config(dir, module, spec, cfg);
    let mut args: Vec<String> = vec![
        "check".to_string(),
        spec_path.to_string_lossy().to_string(),
        "--config".to_string(),
        cfg_path.to_string_lossy().to_string(),
    ];
    for e in extra {
        args.push((*e).to_string());
    }
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    run_tla_parsed(&arg_refs)
}

// ===========================================================================
// V1 — Empty reachable set → VACUOUS (exit 3). Control: ASSUME-only → Success.
// ===========================================================================

#[test]
fn v1_init_false_is_vacuous_exit_3() {
    let dir = TempDir::new("vacuity-v1");
    let spec = r#"
---- MODULE InitFalse ----
VARIABLE x
Init == FALSE
Next == x' = x
Inv == x \in {0, 1}
====
"#;
    let cfg = "INIT Init\nNEXT Next\nINVARIANT Inv\n";
    let (code, stdout, stderr) = check(&dir, "InitFalse", spec, cfg, &["--no-deadlock"]);
    let combined = format!("{stdout}\n{stderr}");
    assert_eq!(
        code, 3,
        "V1 Init==FALSE must exit 3 (VACUOUS). out=\n{combined}"
    );
    assert!(
        combined.to_uppercase().contains("VACUOUS"),
        "expected VACUOUS verdict text. out=\n{combined}"
    );
}

#[test]
fn v1_assume_only_control_is_success_exit_0() {
    // Negative control: a module declaring NO checkable basis (no Init/Next/
    // invariant/property) legitimately admits zero states → Success, NOT Vacuous.
    let dir = TempDir::new("vacuity-v1-control");
    let spec = r#"
---- MODULE AssumeOnly ----
ASSUME TRUE
====
"#;
    let cfg = "";
    let (code, stdout, stderr) = check(&dir, "AssumeOnly", spec, cfg, &[]);
    let combined = format!("{stdout}\n{stderr}");
    assert_eq!(
        code, 0,
        "ASSUME-only control must be Success (exit 0). out=\n{combined}"
    );
    assert!(
        !combined.to_uppercase().contains("VACUOUS"),
        "ASSUME-only control must NOT report VACUOUS. out=\n{combined}"
    );
}

#[test]
fn v1_allow_vacuous_downgrades_to_success_exit_0() {
    // Escape hatch: --allow-vacuous=empty-init downgrades the V1 verdict.
    let dir = TempDir::new("vacuity-v1-allow");
    let spec = r#"
---- MODULE InitFalse2 ----
VARIABLE x
Init == FALSE
Next == x' = x
Inv == x \in {0, 1}
====
"#;
    let cfg = "INIT Init\nNEXT Next\nINVARIANT Inv\n";
    let (code, stdout, stderr) = check(
        &dir,
        "InitFalse2",
        spec,
        cfg,
        &["--no-deadlock", "--allow-vacuous=empty-init"],
    );
    let combined = format!("{stdout}\n{stderr}");
    assert_eq!(
        code, 0,
        "--allow-vacuous=empty-init must downgrade V1 to exit 0. out=\n{combined}"
    );
}

// ===========================================================================
// V2 — Never-enabled (dead) action → default WARNING (exit 0);
//      --strict-vacuity → exit 3. Control: all actions fire → silent.
// ===========================================================================

/// A spec with one always-enabled action (Inc) and one dead action (Dead, guard
/// FALSE). The reachable set is {0,1,2} so V1 does not fire; Dead never fires.
const DEAD_ACTION_SPEC: &str = r#"
---- MODULE DeadAction ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Inc == x < 2 /\ x' = x + 1
Dead == FALSE /\ x' = x
Next == Inc \/ Dead
Inv == x \in 0..2
====
"#;
const DEAD_ACTION_CFG: &str = "INIT Init\nNEXT Next\nINVARIANT Inv\n";

#[test]
fn v2_dead_action_default_is_warning_exit_0() {
    let dir = TempDir::new("vacuity-v2");
    let (code, stdout, stderr) = check(
        &dir,
        "DeadAction",
        DEAD_ACTION_SPEC,
        DEAD_ACTION_CFG,
        &["--no-deadlock"],
    );
    let combined = format!("{stdout}\n{stderr}");
    assert_eq!(
        code, 0,
        "V2 dead action defaults to a non-fatal WARNING (exit 0). out=\n{combined}"
    );
    assert!(
        combined.to_lowercase().contains("dead action"),
        "expected a dead-action warning. out=\n{combined}"
    );
}

#[test]
fn v2_dead_action_strict_is_vacuous_exit_3() {
    let dir = TempDir::new("vacuity-v2-strict");
    let (code, stdout, stderr) = check(
        &dir,
        "DeadAction",
        DEAD_ACTION_SPEC,
        DEAD_ACTION_CFG,
        &["--no-deadlock", "--strict-vacuity"],
    );
    let combined = format!("{stdout}\n{stderr}");
    assert_eq!(
        code, 3,
        "--strict-vacuity promotes the dead-action WARNING to VACUOUS (exit 3). out=\n{combined}"
    );
    assert!(
        combined.to_uppercase().contains("VACUOUS"),
        "expected VACUOUS verdict text under --strict-vacuity. out=\n{combined}"
    );
}

#[test]
fn v2_all_actions_fire_control_is_silent_exit_0() {
    // Control: both actions fire over the reachable set → no dead-action warning.
    let dir = TempDir::new("vacuity-v2-control");
    let spec = r#"
---- MODULE AllFire ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Inc == x < 2 /\ x' = x + 1
Dec == x > 0 /\ x' = x - 1
Next == Inc \/ Dec
Inv == x \in 0..2
====
"#;
    let cfg = "INIT Init\nNEXT Next\nINVARIANT Inv\n";
    // Even under --strict-vacuity the control must pass: no dead action exists.
    let (code, stdout, stderr) = check(
        &dir,
        "AllFire",
        spec,
        cfg,
        &["--no-deadlock", "--strict-vacuity"],
    );
    let combined = format!("{stdout}\n{stderr}");
    assert_eq!(
        code, 0,
        "all-actions-fire control must be silent Success even with --strict-vacuity. out=\n{combined}"
    );
    assert!(
        !combined.to_lowercase().contains("dead action"),
        "control must NOT report a dead action. out=\n{combined}"
    );
}

// ===========================================================================
// V3 — Vacuously-true invariant `P => Q` with P unreachable → WARNING (exit 0);
//      --strict-vacuity → exit 3. Control: constraining invariant → silent.
// ===========================================================================

/// `BadInv == FALSE => (x = 0)` — the antecedent never holds, so the invariant
/// is vacuously true (detected statically as antecedent-folds-to-FALSE).
const VACUOUS_INV_SPEC: &str = r#"
---- MODULE VacuousInv ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Next == x < 2 /\ x' = x + 1
BadInv == FALSE => (x = 0)
====
"#;
const VACUOUS_INV_CFG: &str = "INIT Init\nNEXT Next\nINVARIANT BadInv\n";

#[test]
fn v3_vacuous_invariant_default_is_warning_exit_0() {
    let dir = TempDir::new("vacuity-v3");
    let (code, stdout, stderr) = check(
        &dir,
        "VacuousInv",
        VACUOUS_INV_SPEC,
        VACUOUS_INV_CFG,
        &["--no-deadlock"],
    );
    let combined = format!("{stdout}\n{stderr}");
    assert_eq!(
        code, 0,
        "V3 vacuous invariant defaults to a non-fatal WARNING (exit 0). out=\n{combined}"
    );
    assert!(
        combined.to_lowercase().contains("vacuous")
            && combined.to_lowercase().contains("antecedent"),
        "expected a vacuous-invariant / antecedent warning. out=\n{combined}"
    );
}

#[test]
fn v3_vacuous_invariant_strict_is_vacuous_exit_3() {
    let dir = TempDir::new("vacuity-v3-strict");
    let (code, stdout, stderr) = check(
        &dir,
        "VacuousInv",
        VACUOUS_INV_SPEC,
        VACUOUS_INV_CFG,
        &["--no-deadlock", "--strict-vacuity"],
    );
    let combined = format!("{stdout}\n{stderr}");
    assert_eq!(
        code, 3,
        "--strict-vacuity promotes the vacuous-invariant WARNING to VACUOUS (exit 3). out=\n{combined}"
    );
}

#[test]
fn v3_constraining_invariant_control_is_silent_exit_0() {
    // Control: a genuinely-constraining invariant → no vacuity warning, even
    // under --strict-vacuity.
    let dir = TempDir::new("vacuity-v3-control");
    let spec = r#"
---- MODULE GoodInv ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Next == x < 2 /\ x' = x + 1
GoodInv == x \in 0..2
====
"#;
    let cfg = "INIT Init\nNEXT Next\nINVARIANT GoodInv\n";
    let (code, stdout, stderr) = check(
        &dir,
        "GoodInv",
        spec,
        cfg,
        &["--no-deadlock", "--strict-vacuity"],
    );
    let combined = format!("{stdout}\n{stderr}");
    assert_eq!(
        code, 0,
        "constraining-invariant control must be silent Success even with --strict-vacuity. out=\n{combined}"
    );
    assert!(
        !combined.to_lowercase().contains("vacuous"),
        "control must NOT report a vacuous invariant. out=\n{combined}"
    );
}
