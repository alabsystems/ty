// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;

// InitMode helper method tests - Part of #524 self-audit
//
// Note: InitMode::resolve() is not tested here because it reads actual
// environment variables (TY_FORCE_AY, TY_USE_AY). Testing env var logic
// in parallel test suites requires isolation (e.g., serial_test crate or
// temp_env) to avoid tests affecting each other. The logic is simple enough
// (3-way if/else) that the risk of bugs is low. Integration tests exercise
// the full path via TY_FORCE_AY/TY_USE_AY env vars.

// Part of #2757: ay_decision and should_skip_analysis are gated behind
// cfg(feature = "ay") in production code; tests must match.
#[cfg(feature = "ay")]
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_init_mode_ay_decision_brute_force() {
    // BruteForce should never try ay
    assert_eq!(InitMode::BruteForce.ay_decision(true), (false, false));
    assert_eq!(InitMode::BruteForce.ay_decision(false), (false, false));
}

#[cfg(feature = "ay")]
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_init_mode_ay_decision_auto() {
    // Auto should follow analysis recommendation
    assert_eq!(InitMode::Auto.ay_decision(true), (false, true));
    assert_eq!(InitMode::Auto.ay_decision(false), (false, false));
}

#[cfg(feature = "ay")]
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_init_mode_ay_decision_force_ay() {
    // ForceAY should always try ay
    assert_eq!(InitMode::ForceAY.ay_decision(true), (true, true));
    assert_eq!(InitMode::ForceAY.ay_decision(false), (true, true));
}

#[cfg(feature = "ay")]
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_init_mode_should_skip_analysis() {
    assert!(InitMode::BruteForce.should_skip_analysis());
    assert!(!InitMode::Auto.should_skip_analysis());
    assert!(!InitMode::ForceAY.should_skip_analysis());
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_config_default_init_mode() {
    let config = Config::default();
    assert_eq!(config.init_mode, InitMode::Auto);
}
