// Copyright 2026 Andrew Yates.
// Author: Andrew Yates
// Licensed under the Apache License, Version 2.0

//! Compare test results between commits to identify regressions.
//! Rust port of scripts/compare_test_results.py

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;

/// Command-line arguments for the test-results comparison command.
#[derive(Parser, Debug)]
pub struct TestResultsCompareArgs {
    /// Compare specific commits (provide two positional arguments)
    #[arg(num_args = 0..=2)]
    pub commits: Vec<String>,

    /// Compare against baseline
    #[arg(long)]
    pub baseline: bool,
}

#[derive(Deserialize, Debug, Clone)]
struct GitInfo {
    #[serde(default)]
    commit: String,
    #[serde(default)]
    short: String,
}

#[derive(Deserialize, Debug, Clone)]
struct SpecResult {
    states: Option<i64>,
    expected_states: Option<i64>,
    #[serde(default = "default_status")]
    status: String,
}

fn default_status() -> String {
    "unknown".to_string()
}

#[derive(Deserialize, Debug, Clone)]
struct HistoryEntry {
    git: Option<GitInfo>,
    results: HashMap<String, SpecResult>,
}

#[derive(Deserialize, Debug)]
struct BaselineFile {
    specs: HashMap<String, SpecResult>,
}

fn load_history() -> Result<Vec<HistoryEntry>> {
    let path = Path::new("tests/tlc_comparison/ty_results_history.json");
    if !path.exists() {
        println!("No history file: {}", path.display());
        return Ok(Vec::new());
    }
    let content = fs::read_to_string(path)?;
    serde_json::from_str(&content).context("Failed to parse history JSON")
}

fn load_baseline() -> Result<HashMap<String, SpecResult>> {
    let path = Path::new("tests/tlc_comparison/spec_baseline.json");
    if !path.exists() {
        println!("No baseline file: {}", path.display());
        return Ok(HashMap::new());
    }
    let content = fs::read_to_string(path)?;
    let parsed: BaselineFile =
        serde_json::from_str(&content).context("Failed to parse baseline JSON")?;
    Ok(parsed.specs)
}

fn find_run_by_commit(history: &[HistoryEntry], commit: &str) -> Option<HistoryEntry> {
    for entry in history.iter().rev() {
        if let Some(git) = &entry.git {
            if git.commit.starts_with(commit) || git.short == commit {
                return Some(entry.clone());
            }
        }
    }
    None
}

fn compare_runs(old: &HistoryEntry, new: &HistoryEntry, old_label: &str, new_label: &str) {
    let mut regressions = Vec::new();
    let mut new_failures = Vec::new();
    let mut new_passes = Vec::new();

    let mut all_specs = BTreeSet::new();
    for k in old.results.keys() {
        all_specs.insert(k.clone());
    }
    for k in new.results.keys() {
        all_specs.insert(k.clone());
    }

    for name in all_specs {
        let old_r = old.results.get(&name);
        let new_r = new.results.get(&name);

        let old_states = old_r.and_then(|r| r.states.or(r.expected_states));
        let new_states = new_r.and_then(|r| r.states);
        let old_status = old_r.map(|r| r.status.as_str()).unwrap_or("unknown");
        let new_status = new_r.map(|r| r.status.as_str()).unwrap_or("unknown");

        if let (Some(os), Some(ns)) = (old_states, new_states) {
            if os != ns {
                regressions.push((name.clone(), os, ns));
            }
        }

        if old_status == "pass" && new_status != "pass" {
            new_failures.push((name.clone(), old_status.to_string(), new_status.to_string()));
        } else if old_status != "pass" && new_status == "pass" {
            new_passes.push((name.clone(), old_status.to_string(), new_status.to_string()));
        }
    }

    println!("{:=<70}", "");
    println!("COMPARISON: {} -> {}", old_label, new_label);
    println!("{:=<70}\n", "");

    if !regressions.is_empty() {
        println!("STATE COUNT REGRESSIONS ({}):", regressions.len());
        for (name, old_s, new_s) in &regressions {
            let diff = new_s - old_s;
            let pct = if *old_s != 0 {
                (diff as f64 / *old_s as f64) * 100.0
            } else {
                0.0
            };
            println!(
                "  {}: {} -> {} ({:+} , {:+.1}%)",
                name, old_s, new_s, diff, pct
            );
        }
        println!();
    }

    if !new_failures.is_empty() {
        println!("NEW FAILURES ({}):", new_failures.len());
        for (name, old_s, new_s) in &new_failures {
            println!("  {}: {} -> {}", name, old_s, new_s);
        }
        println!();
    }

    if !new_passes.is_empty() {
        println!("NEW PASSES ({}):", new_passes.len());
        for (name, old_s, new_s) in &new_passes {
            println!("  {}: {} -> {}", name, old_s, new_s);
        }
        println!();
    }

    let old_pass = old.results.values().filter(|r| r.status == "pass").count();
    let new_pass = new.results.values().filter(|r| r.status == "pass").count();

    println!("SUMMARY:");
    println!("  {}: {} passing", old_label, old_pass);
    println!("  {}: {} passing", new_label, new_pass);
    println!("  Change: {:+}", new_pass as i64 - old_pass as i64);

    if !regressions.is_empty() {
        println!(
            "\n  ** {} STATE COUNT REGRESSIONS - INVESTIGATE **",
            regressions.len()
        );
    }
}

/// Compares stored test-result history (against another commit or the TLC
/// baseline) and prints the differences.
///
/// # Errors
///
/// Returns an error if the test history or baseline files cannot be read or
/// parsed.
pub fn run(args: TestResultsCompareArgs) -> Result<()> {
    if args.baseline {
        let history = load_history()?;
        let baseline_specs = load_baseline()?;

        if history.is_empty() {
            println!("No test history to compare");
            return Ok(());
        }
        if baseline_specs.is_empty() {
            println!("No baseline to compare against");
            return Ok(());
        }

        let latest = history.last().unwrap().clone();
        let baseline_run = HistoryEntry {
            git: Some(GitInfo {
                short: "TLC-baseline".to_string(),
                commit: String::new(),
            }),
            results: baseline_specs,
        };

        let new_label = latest
            .git
            .as_ref()
            .map(|g| g.short.as_str())
            .unwrap_or("latest");
        compare_runs(&baseline_run, &latest, "TLC-baseline", new_label);
    } else if args.commits.len() == 2 {
        let history = load_history()?;
        let old = find_run_by_commit(&history, &args.commits[0]);
        let new = find_run_by_commit(&history, &args.commits[1]);

        if old.is_none() {
            println!("No run found for commit: {}", args.commits[0]);
            return Ok(());
        }
        if new.is_none() {
            println!("No run found for commit: {}", args.commits[1]);
            return Ok(());
        }

        compare_runs(
            old.as_ref().unwrap(),
            new.as_ref().unwrap(),
            &args.commits[0],
            &args.commits[1],
        );
    } else {
        let history = load_history()?;
        if history.len() < 2 {
            println!("Need at least 2 runs to compare");
            if !history.is_empty() {
                println!("Only have {} run(s)", history.len());
            }
            return Ok(());
        }

        let old = &history[history.len() - 2];
        let new = &history[history.len() - 1];

        let old_label = old
            .git
            .as_ref()
            .map(|g| g.short.as_str())
            .unwrap_or("previous");
        let new_label = new
            .git
            .as_ref()
            .map(|g| g.short.as_str())
            .unwrap_or("latest");

        compare_runs(old, new, old_label, new_label);
    }

    Ok(())
}
