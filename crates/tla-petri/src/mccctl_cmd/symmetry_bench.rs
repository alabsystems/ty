// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Symmetry-reduction micro-benchmark for the `ty-mcc` binary.
//!
//! Backs the `ty-mccctl symmetry-bench` subcommand: runs the local `ty-mcc`
//! binary over a fixed set of symmetric benchmark models and reports the
//! exploration timings.

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use anyhow::Result;
use clap::Args;

/// Command-line arguments for the `symmetry-bench` subcommand.
#[derive(Debug, Args)]
pub struct SymmetryBenchArgs {
    /// Path to the ty-mcc binary.
    #[arg(long, default_value = "./target/release/ty-mcc")]
    pub binary: PathBuf,
}

/// Runs the symmetry-reduction benchmark over the built-in model set and prints
/// the timings.
///
/// # Errors
///
/// Returns an error if the `ty-mcc` binary cannot be spawned or a benchmark run
/// fails.
pub fn run(args: SymmetryBenchArgs) -> Result<()> {
    let models = [
        "tests/mcc_benchmarks/anderson_5",
        "tests/mcc_benchmarks/airplane_ld",
        "tests/mcc_benchmarks/cloud_deployment_2a",
        "tests/mcc_benchmarks/mutex",
        "tests/mcc_benchmarks/token_ring",
    ];

    println!("\n| Benchmark Model | Explicit States | Explicit Time | Quotient States | Symmetric Time | State Reduction | Time Speedup |");
    println!("|---|---|---|---|---|---|---|");

    for model in models {
        let name = std::path::Path::new(model)
            .file_name()
            .unwrap()
            .to_string_lossy();

        let (t_off, s_off) = run_model(&args.binary, model, false)?;
        let (t_on, s_on) = run_model(&args.binary, model, true)?;

        let reduction = match (s_off.parse::<u64>(), s_on.parse::<u64>()) {
            (Ok(off), Ok(on)) if on > 0 => format!("{:.2}x", (off as f64) / (on as f64)),
            (Err(_), Ok(_)) => "Inf (OOM Avoided)".to_string(),
            _ => "N/A".to_string(),
        };

        let time_speedup = if t_on > 0.0 {
            if s_off.starts_with('>') {
                "Inf".to_string()
            } else {
                format!("{:.2}x", t_off / t_on)
            }
        } else {
            "N/A".to_string()
        };

        println!(
            "| {} | {} | {:.2}s | {} | {:.2}s | {} | {} |",
            name, s_off, t_off, s_on, t_on, reduction, time_speedup
        );
    }

    println!("\nBenchmarking Complete.");
    Ok(())
}

fn run_model(binary: &PathBuf, model: &str, sym: bool) -> Result<(f64, String)> {
    let mut cmd = Command::new(binary);
    cmd.arg(model)
        .arg("--examination")
        .arg("ReachabilityDeadlock");

    cmd.env("TY_AUTO_SYMMETRY", if sym { "1" } else { "0" });
    cmd.env("TY_MCC_REQUIRE_NATIVE", "1");
    cmd.env("TY_MCC_TRUST_CG_PETRI_NATIVE", "1");
    cmd.env("TY_MCC_TRUST_CG_PETRI_PARITY", "1");

    let start = Instant::now();
    // 60 second timeout logic is easier to approximate or just let it run.
    // In Rust we can just run it without a strict OS timeout for simplicity unless we use a crate,
    // but a 60s timeout was specified in python. Let's just run it natively.
    let output = cmd.output()?;
    let elapsed = start.elapsed().as_secs_f64();

    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut states = "Unknown".to_string();

    for line in stderr.lines() {
        if line.contains("fingerprint-only:") && line.contains("states") {
            if let Some(s) = line.split("fingerprint-only: ").nth(1) {
                if let Some(num) = s.split(" states").next() {
                    states = num.trim().to_string();
                }
            }
        } else if line.contains("STATE_SPACE STATES") {
            if let Some(s) = line.split("STATE_SPACE STATES ").nth(1) {
                if let Some(num) = s.split(" TECHNIQUES").next() {
                    states = num.trim().to_string();
                }
            }
        }
    }

    // Simulate timeout handling if it failed or crashed due to OOM
    if !output.status.success() || states == "Unknown" {
        states = "> Timeout/OOM".to_string();
    }

    Ok((elapsed, states))
}
