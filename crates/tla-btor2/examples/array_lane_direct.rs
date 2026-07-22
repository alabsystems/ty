// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Engine-direct array-lane runner (the phase-3 L2 sweep entry point):
//! runs the lazy-array BMC, k-induction, and IC3 engines on one BTOR2 file
//! with a per-engine wall budget, free of the CLI's leading-slice clamp.
//!
//! Usage: `array_lane_direct <net.btor2> [budget_secs]`
//!
//! Output: one line per engine, machine-parseable:
//! `BMC UNSAFE@d | BMC BOUNDED=k | BMC DECLINED reason` etc. Verdict rules
//! are the lanes' own (replay-gated SAT, LRAT-gated unbounded-safe).

use std::time::Duration;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: array_lane_direct <net.btor2> [budget_secs]");
        std::process::exit(2);
    }
    let budget_secs: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(120);
    let budget = Some(Duration::from_secs(budget_secs));

    let source = std::fs::read_to_string(&args[1]).expect("read net");
    let program = match tla_btor2::parse_btor2(&source) {
        Ok(p) => p,
        Err(e) => {
            println!("PARSE DECLINED {e}");
            return;
        }
    };

    let bmc = tla_btor2::check_array_bmc(
        &program,
        &tla_btor2::ArrayBmcConfig {
            time_budget: budget,
            ..tla_btor2::ArrayBmcConfig::default()
        },
    );
    match &bmc {
        tla_btor2::ArrayBmcOutcome::Unsafe { depth, .. } => println!("BMC UNSAFE@{depth}"),
        tla_btor2::ArrayBmcOutcome::BoundedNoCex { depth_reached } => {
            println!("BMC BOUNDED={depth_reached}")
        }
        tla_btor2::ArrayBmcOutcome::Declined { reason } => println!("BMC DECLINED {reason}"),
    }

    let kind = tla_btor2::check_array_kinduction(
        &program,
        &tla_btor2::ArrayKindConfig {
            time_budget: budget,
            ..tla_btor2::ArrayKindConfig::default()
        },
    );
    match &kind {
        tla_btor2::ArrayKindOutcome::ProvedSafe { k } => println!("KIND PROVED k={k}"),
        tla_btor2::ArrayKindOutcome::Unsafe { depth, .. } => println!("KIND UNSAFE@{depth}"),
        tla_btor2::ArrayKindOutcome::BoundedNoCex { depth_reached } => {
            println!("KIND BOUNDED={depth_reached}")
        }
        tla_btor2::ArrayKindOutcome::Declined { reason } => println!("KIND DECLINED {reason}"),
    }

    let ic3 = tla_btor2::check_array_ic3(
        &program,
        &tla_btor2::ArrayIc3Config {
            time_budget: budget,
            ..tla_btor2::ArrayIc3Config::default()
        },
    );
    match &ic3 {
        tla_btor2::ArrayIc3Outcome::ProvedSafe {
            converged_at,
            invariant,
            tier,
        } => println!(
            "IC3 PROVED level={converged_at} clauses={} probes={} tier={tier:?}",
            invariant.clauses.len(),
            invariant.probes.len()
        ),
        tla_btor2::ArrayIc3Outcome::Unsafe { depth, .. } => println!("IC3 UNSAFE@{depth}"),
        tla_btor2::ArrayIc3Outcome::BoundedNoCex { frames_completed } => {
            println!("IC3 BOUNDED={frames_completed}")
        }
        tla_btor2::ArrayIc3Outcome::Declined { reason } => println!("IC3 DECLINED {reason}"),
    }
}
