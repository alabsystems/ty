// Copyright 2026 Andrew Yates.
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Summarize dhat heap profiles by allocation family and hot stack.
//! Rust port of scripts/dhat_summary.py.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;

/// Command-line arguments for the dhat-profile summarizer.
#[derive(Parser, Debug)]
pub struct DhatSummaryArgs {
    /// Path to dhat-heap.json
    pub profile: PathBuf,

    /// Number of families/stacks to show
    #[arg(long, default_value_t = 10)]
    pub top: usize,

    /// Maximum number of frames to print per exemplar stack
    #[arg(long, default_value_t = 6)]
    pub stack_limit: usize,
}

#[derive(Deserialize, Debug)]
struct DhatProfile {
    #[serde(default)]
    cmd: String,
    #[serde(default)]
    ftbl: Vec<String>,
    #[serde(default)]
    pps: Vec<DhatProgramPoint>,
}

#[derive(Deserialize, Debug)]
struct DhatProgramPoint {
    #[serde(default)]
    tbk: u64,
    #[serde(default)]
    tb: u64,
    #[serde(default)]
    fs: Vec<usize>,
}

#[derive(Debug, Clone)]
struct StackSample {
    tbk: u64,
    tb: u64,
    frames: Vec<String>,
    family: String,
}

#[derive(Debug)]
struct FamilySummary {
    family: String,
    tbk: u64,
    tb: u64,
    samples: usize,
    exemplar: Vec<String>,
}

const WORKSPACE_PREFIXES: &[&str] = &["tla_", "ty"];
const RUNTIME_PREFIXES: &[&str] = &[
    "[root]",
    "0x",
    "__",
    "<alloc::",
    "<core::",
    "<std::",
    "alloc::",
    "core::",
    "std::",
    "dhat::",
    "mimalloc::",
];

fn normalize_frame(frame: &str) -> &str {
    if frame == "[root]" {
        return frame;
    }
    if frame.starts_with("0x") {
        if let Some((_, remainder)) = frame.split_once(": ") {
            return remainder;
        }
    }
    frame
}

fn frame_symbol(frame: &str) -> &str {
    let normalized = normalize_frame(frame);
    if normalized == "[root]" {
        return normalized;
    }
    if let Some((symbol, _)) = normalized.split_once(" (") {
        return symbol;
    }
    normalized
}

fn is_workspace_symbol(symbol: &str) -> bool {
    WORKSPACE_PREFIXES.iter().any(|&p| symbol.starts_with(p))
}

fn is_runtime_symbol(symbol: &str) -> bool {
    RUNTIME_PREFIXES.iter().any(|&p| symbol.starts_with(p))
}

fn classify_family(frames: &[String]) -> String {
    let symbols: Vec<&str> = frames.iter().map(|f| frame_symbol(f)).collect();
    for &sym in &symbols {
        if is_workspace_symbol(sym) {
            return sym.to_string();
        }
    }
    for &sym in &symbols {
        if !is_runtime_symbol(sym) {
            return sym.to_string();
        }
    }
    if let Some(&first) = symbols.first() {
        first.to_string()
    } else {
        "[no frames]".to_string()
    }
}

fn trim_stack(frames: &[String], limit: usize, focus_symbol: Option<&str>) -> Vec<String> {
    let symbols: Vec<&str> = frames.iter().map(|f| frame_symbol(f)).collect();
    let mut first_relevant = 0;

    if let Some(focus) = focus_symbol {
        if let Some(idx) = symbols.iter().position(|&s| s == focus) {
            first_relevant = idx;
        } else if let Some(idx) = symbols.iter().position(|&s| !is_runtime_symbol(s)) {
            first_relevant = idx;
        }
    } else if let Some(idx) = symbols.iter().position(|&s| !is_runtime_symbol(s)) {
        first_relevant = idx;
    }

    let mut trimmed: Vec<String> = symbols
        .into_iter()
        .skip(first_relevant)
        .map(|s| s.to_string())
        .collect();
    if limit > 0 && trimmed.len() > limit {
        trimmed.truncate(limit);
    }
    trimmed
}

/// Reads a dhat heap profile and prints a human-readable allocation summary.
///
/// # Errors
///
/// Returns an error if the profile file cannot be read or does not parse as
/// the expected dhat-heap JSON schema.
pub fn run(args: DhatSummaryArgs) -> Result<()> {
    let content = fs::read_to_string(&args.profile)
        .with_context(|| format!("Failed to read {}", args.profile.display()))?;
    let data: DhatProfile =
        serde_json::from_str(&content).with_context(|| "Failed to parse JSON")?;

    let mut samples = Vec::new();
    for pp in &data.pps {
        let mut frames = Vec::new();
        for &f_idx in &pp.fs {
            if let Some(f_str) = data.ftbl.get(f_idx) {
                frames.push(f_str.clone());
            }
        }
        let family = classify_family(&frames);
        samples.push(StackSample {
            tbk: pp.tbk,
            tb: pp.tb,
            frames,
            family,
        });
    }

    let mut family_totals: HashMap<String, FamilySummary> = HashMap::new();
    let mut best_tbk_map: HashMap<String, u64> = HashMap::new();

    for sample in &samples {
        let entry = family_totals
            .entry(sample.family.clone())
            .or_insert_with(|| FamilySummary {
                family: sample.family.clone(),
                tbk: 0,
                tb: 0,
                samples: 0,
                exemplar: Vec::new(),
            });
        entry.tbk += sample.tbk;
        entry.tb += sample.tb;
        entry.samples += 1;

        let best_tbk = best_tbk_map.entry(sample.family.clone()).or_insert(0);
        if sample.tbk >= *best_tbk {
            entry.exemplar = sample.frames.clone();
            *best_tbk = sample.tbk;
        }
    }

    let mut top_families: Vec<FamilySummary> = family_totals.into_values().collect();
    top_families.sort_by(|a, b| {
        b.tbk
            .cmp(&a.tbk)
            .then(b.tb.cmp(&a.tb))
            .then(a.family.cmp(&b.family))
    });
    if args.top > 0 && top_families.len() > args.top {
        top_families.truncate(args.top);
    }

    let mut top_stacks = samples.clone();
    top_stacks.sort_by(|a, b| b.tbk.cmp(&a.tbk).then(b.tb.cmp(&a.tb)));
    if args.top > 0 && top_stacks.len() > args.top {
        top_stacks.truncate(args.top);
    }

    let total_blocks: u64 = samples.iter().map(|s| s.tbk).sum();
    let total_bytes: u64 = samples.iter().map(|s| s.tb).sum();
    let total_samples = samples.len();

    println!("Command: {}", data.cmd);
    println!(
        "Totals: {} blocks, {} bytes, {} program points\n",
        total_blocks, total_bytes, total_samples
    );

    println!("Top families by allocation blocks:");
    for (i, family) in top_families.iter().enumerate() {
        let stack_text =
            trim_stack(&family.exemplar, args.stack_limit, Some(&family.family)).join(" -> ");
        println!(
            "{}. {} | blocks={} bytes={} samples={}",
            i + 1,
            family.family,
            family.tbk,
            family.tb,
            family.samples
        );
        if !stack_text.is_empty() {
            println!("   stack: {}", stack_text);
        }
    }
    println!("\nTop exact stacks by allocation blocks:");
    for (i, stack) in top_stacks.iter().enumerate() {
        let stack_text =
            trim_stack(&stack.frames, args.stack_limit, Some(&stack.family)).join(" -> ");
        println!(
            "{}. blocks={} bytes={} family={}",
            i + 1,
            stack.tbk,
            stack.tb,
            stack.family
        );
        if !stack_text.is_empty() {
            println!("   stack: {}", stack_text);
        }
    }

    Ok(())
}
