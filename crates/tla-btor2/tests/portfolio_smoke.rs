// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Smoke tests for the HWMCC portfolio pipeline (COI + BMC + CHC).
//!
//! Validates the portfolio strategy on hand-crafted programs and verifies
//! that COI reduction correctly identifies irrelevant state variables.
//!
//! Run with: cargo test -p tla-btor2 --test portfolio_smoke

use std::time::Duration;

use tla_btor2::{parse, PortfolioConfig};

/// Test portfolio on a simple counter that counts to 3 (shallow bug).
/// The BMC phase should catch this quickly.
#[test]
fn test_portfolio_counter_to_3() {
    let src = "\
1 sort bitvec 8
2 sort bitvec 1
3 zero 1
4 state 1 count
5 init 1 4 3
6 one 1
7 add 1 4 6
8 next 1 4 7
9 constd 1 3
10 eq 2 4 9
11 bad 10
";
    let program = parse(src).expect("should parse");
    let config = PortfolioConfig {
        time_budget: Some(Duration::from_secs(30)),
        enable_coi: true,
        enable_simplify: true,
        enable_bmc: true,
        bmc_budget_fraction: 0.3,
        bmc_max_depth: 10,
        verbose: false,
    };

    let (results, stats) =
        tla_btor2::check_btor2_portfolio(&program, &config).expect("should succeed");
    assert_eq!(results.len(), 1);
    match &results[0] {
        tla_btor2::Btor2CheckResult::Sat { trace, .. } => {
            assert!(!trace.is_empty(), "should have counterexample steps");
        }
        other => panic!("expected Sat, got: {:?}", other),
    }
    // No COI reduction (only 1 state, all relevant).
    assert_eq!(stats.states_before_coi, 1);
    assert_eq!(stats.states_after_coi, 1);
}

/// Test that a trivially safe program (state=0 always, bad=state!=0) is verified.
#[test]
fn test_portfolio_trivially_safe() {
    let src = "\
1 sort bitvec 8
2 sort bitvec 1
3 zero 1
4 state 1 s
5 init 1 4 3
6 next 1 4 3
7 one 1
8 eq 2 4 7
9 bad 8
";
    let program = parse(src).expect("should parse");
    let config = PortfolioConfig {
        time_budget: Some(Duration::from_secs(30)),
        ..PortfolioConfig::default()
    };

    let (results, _stats) =
        tla_btor2::check_btor2_portfolio(&program, &config).expect("should succeed");
    assert_eq!(results.len(), 1);
    match &results[0] {
        tla_btor2::Btor2CheckResult::Unsat => {} // Expected: safe
        other => panic!("expected Unsat (safe), got: {:?}", other),
    }
}

/// Test COI reduction: program with 2 states but only 1 affects the bad property.
#[test]
fn test_portfolio_coi_eliminates_irrelevant_state() {
    // State x starts at 0, increments by 1.
    // State y starts at 0, stays at 0 (completely irrelevant).
    // Bad: x == 5
    let src = "\
1 sort bitvec 8
2 sort bitvec 1
3 zero 1
4 state 1 x
5 init 1 4 3
6 one 1
7 add 1 4 6
8 next 1 4 7
9 state 1 y
10 init 1 9 3
11 next 1 9 3
12 constd 1 5
13 eq 2 4 12
14 bad 13
";
    let program = parse(src).expect("should parse");
    assert_eq!(program.num_states, 2);

    let config = PortfolioConfig {
        time_budget: Some(Duration::from_secs(30)),
        enable_coi: true,
        ..PortfolioConfig::default()
    };

    let (results, stats) =
        tla_btor2::check_btor2_portfolio(&program, &config).expect("should succeed");
    assert_eq!(results.len(), 1);

    // COI should eliminate state y.
    assert_eq!(stats.states_before_coi, 2);
    assert_eq!(stats.states_after_coi, 1, "COI should eliminate state y");

    match &results[0] {
        tla_btor2::Btor2CheckResult::Sat { trace, .. } => {
            assert!(!trace.is_empty());
        }
        other => panic!("expected Sat, got: {:?}", other),
    }
}

/// Test portfolio with no bad properties.
#[test]
fn test_portfolio_no_properties() {
    let src = "\
1 sort bitvec 1
2 state 1 s
";
    let program = parse(src).expect("should parse");
    let config = PortfolioConfig::default();

    let (results, _stats) =
        tla_btor2::check_btor2_portfolio(&program, &config).expect("should succeed");
    assert!(results.is_empty());
}

/// Test that disabling COI still produces correct results.
#[test]
fn test_portfolio_coi_disabled() {
    let src = "\
1 sort bitvec 8
2 sort bitvec 1
3 zero 1
4 state 1 count
5 init 1 4 3
6 one 1
7 add 1 4 6
8 next 1 4 7
9 constd 1 3
10 eq 2 4 9
11 bad 10
";
    let program = parse(src).expect("should parse");
    let config = PortfolioConfig {
        time_budget: Some(Duration::from_secs(30)),
        enable_coi: false,
        ..PortfolioConfig::default()
    };

    let (results, stats) =
        tla_btor2::check_btor2_portfolio(&program, &config).expect("should succeed");
    assert_eq!(results.len(), 1);
    match &results[0] {
        tla_btor2::Btor2CheckResult::Sat { .. } => {} // Expected
        other => panic!("expected Sat, got: {:?}", other),
    }
    // No COI reduction when disabled.
    assert_eq!(stats.states_before_coi, stats.states_after_coi);
}

/// SOUNDNESS regression (gate-expression init): `init s = concat(i, i)`
/// constrains the initial state to s ∈ {00, 11}; `next s = s` holds it; bad =
/// (s == 01) is unreachable — the TRUE verdict is unsat. Bit-blasting used to
/// over-approximate this init as a nondeterministic reset (admitting phantom
/// initial states 01/10) and reported a spurious `sat`; the net is now
/// bit-blast-INELIGIBLE and must be proven safe by the word-level lane, which
/// encodes the init value exactly.
#[test]
fn test_portfolio_gate_expression_init_safe_is_unsat() {
    let src = "\
1 sort bitvec 1
2 sort bitvec 2
3 input 1 i
4 state 2 s
5 concat 2 3 3
6 init 2 4 5
7 next 2 4 4
8 constd 2 1
9 eq 1 4 8
10 bad 9
";
    let program = parse(src).expect("should parse");
    assert!(
        tla_btor2::bitblast_eligible(&program, 32).is_err(),
        "gate-expression init must be bit-blast-ineligible (routes to the word-level lane)"
    );
    let config = PortfolioConfig {
        time_budget: Some(Duration::from_secs(30)),
        ..PortfolioConfig::default()
    };

    let (results, _stats) =
        tla_btor2::check_btor2_portfolio(&program, &config).expect("should succeed");
    assert_eq!(results.len(), 1);
    match &results[0] {
        tla_btor2::Btor2CheckResult::Unsat => {} // Expected: genuinely safe
        other => panic!("expected Unsat (safe), got: {:?}", other),
    }
}

/// Companion to the soundness regression above: the SAME gate-expression init
/// (s ∈ {00, 11}), but bad = (s == 11) — genuinely reachable from the initial
/// state where input i = 1. The word-level lane must still find the real
/// counterexample (sat); fail-closed routing must not cost completeness.
#[test]
fn test_portfolio_gate_expression_init_violable_is_sat() {
    let src = "\
1 sort bitvec 1
2 sort bitvec 2
3 input 1 i
4 state 2 s
5 concat 2 3 3
6 init 2 4 5
7 next 2 4 4
8 constd 2 3
9 eq 1 4 8
10 bad 9
";
    let program = parse(src).expect("should parse");
    assert!(
        tla_btor2::bitblast_eligible(&program, 32).is_err(),
        "gate-expression init must be bit-blast-ineligible (routes to the word-level lane)"
    );
    let config = PortfolioConfig {
        time_budget: Some(Duration::from_secs(30)),
        ..PortfolioConfig::default()
    };

    let (results, _stats) =
        tla_btor2::check_btor2_portfolio(&program, &config).expect("should succeed");
    assert_eq!(results.len(), 1);
    match &results[0] {
        tla_btor2::Btor2CheckResult::Sat { .. } => {} // Expected: genuinely violable
        other => panic!("expected Sat (genuine violation), got: {:?}", other),
    }
}
