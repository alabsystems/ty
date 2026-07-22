// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Summarize dhat heap profiles by allocation family and hot stack.
//!
//! Rust port of `scripts/dhat_summary.py`. Reads a dhat-heap.json profile,
//! groups program-point samples by workspace family symbol, and prints
//! stable text output suitable for reports and issue comments.

use std::cmp::Ordering;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::Value;

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

#[derive(Parser, Debug)]
#[command(
    name = "ty-dhat-summary",
    about = "Summarize a dhat heap profile by allocation family and hot stack"
)]
struct Cli {
    /// Path to `dhat-heap.json`.
    profile: PathBuf,
    /// Number of families/stacks to show.
    #[arg(long, default_value_t = 10)]
    top: usize,
    /// Maximum number of frames to print per exemplar stack (0 = unlimited).
    #[arg(long = "stack-limit", default_value_t = 6)]
    stack_limit: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct StackSample {
    tbk: i64,
    tb: i64,
    frames: Vec<String>,
    family: String,
}

#[derive(Clone, Debug, PartialEq)]
struct FamilySummary {
    family: String,
    tbk: i64,
    tb: i64,
    samples: usize,
    exemplar: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
struct ProfileSummary {
    command: String,
    total_blocks: i64,
    total_bytes: i64,
    total_samples: usize,
    top_families: Vec<FamilySummary>,
    top_stacks: Vec<StackSample>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if cli.top == 0 {
        eprintln!("error: --top must be positive");
        return ExitCode::from(2);
    }
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err:#}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: &Cli) -> Result<()> {
    let text = fs::read_to_string(&cli.profile)
        .with_context(|| format!("reading {}", cli.profile.display()))?;
    let data: Value = serde_json::from_str(&text)
        .with_context(|| format!("parsing {}", cli.profile.display()))?;
    let summary = summarize_profile(&data, cli.top)?;
    println!("{}", render_summary(&summary, cli.stack_limit));
    Ok(())
}

fn normalize_frame(frame: &str) -> &str {
    if frame == "[root]" {
        return frame;
    }
    if frame.starts_with("0x") {
        if let Some(idx) = frame.find(": ") {
            return &frame[idx + 2..];
        }
    }
    frame
}

fn frame_symbol(frame: &str) -> &str {
    let normalized = normalize_frame(frame);
    if normalized == "[root]" {
        return normalized;
    }
    match normalized.find(" (") {
        Some(idx) => &normalized[..idx],
        None => normalized,
    }
}

fn is_workspace_symbol(symbol: &str) -> bool {
    WORKSPACE_PREFIXES.iter().any(|p| symbol.starts_with(p))
}

fn is_runtime_symbol(symbol: &str) -> bool {
    RUNTIME_PREFIXES.iter().any(|p| symbol.starts_with(p))
}

fn classify_family(frames: &[String]) -> String {
    let symbols: Vec<&str> = frames.iter().map(|f| frame_symbol(f)).collect();
    for sym in &symbols {
        if is_workspace_symbol(sym) {
            return (*sym).to_string();
        }
    }
    for sym in &symbols {
        if !is_runtime_symbol(sym) {
            return (*sym).to_string();
        }
    }
    symbols
        .first()
        .map(|s| (*s).to_string())
        .unwrap_or_else(|| "[no frames]".to_string())
}

fn stack_from_pp(pp: &Value, ftbl: &[String]) -> StackSample {
    let tbk = pp.get("tbk").and_then(|v| v.as_i64()).unwrap_or(0);
    let tb = pp.get("tb").and_then(|v| v.as_i64()).unwrap_or(0);
    let frames: Vec<String> = pp
        .get("fs")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|idx| idx.as_i64())
                .filter_map(|idx| {
                    let i = usize::try_from(idx).ok()?;
                    ftbl.get(i).cloned()
                })
                .collect()
        })
        .unwrap_or_default();
    let family = classify_family(&frames);
    StackSample {
        tbk,
        tb,
        frames,
        family,
    }
}

fn summarize_profile(data: &Value, top: usize) -> Result<ProfileSummary> {
    let ftbl: Vec<String> = data
        .get("ftbl")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|v| match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .collect()
        })
        .unwrap_or_default();
    let pps = data
        .get("pps")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let mut samples: Vec<StackSample> = pps.iter().map(|pp| stack_from_pp(pp, &ftbl)).collect();

    let command = data
        .get("cmd")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let total_blocks: i64 = samples.iter().map(|s| s.tbk).sum();
    let total_bytes: i64 = samples.iter().map(|s| s.tb).sum();
    let total_samples = samples.len();

    // Aggregate by family.
    let mut family_map: std::collections::BTreeMap<String, FamilyAgg> =
        std::collections::BTreeMap::new();
    for sample in &samples {
        let entry = family_map
            .entry(sample.family.clone())
            .or_insert_with(|| FamilyAgg {
                tbk: 0,
                tb: 0,
                samples: 0,
                exemplar: sample.frames.clone(),
                best_tbk: i64::MIN,
            });
        entry.tbk += sample.tbk;
        entry.tb += sample.tb;
        entry.samples += 1;
        if sample.tbk > entry.best_tbk {
            entry.exemplar = sample.frames.clone();
            entry.best_tbk = sample.tbk;
        }
    }
    let mut top_families: Vec<FamilySummary> = family_map
        .into_iter()
        .map(|(family, agg)| FamilySummary {
            family,
            tbk: agg.tbk,
            tb: agg.tb,
            samples: agg.samples,
            exemplar: agg.exemplar,
        })
        .collect();
    top_families.sort_by(|a, b| {
        b.tbk
            .cmp(&a.tbk)
            .then_with(|| b.tb.cmp(&a.tb))
            .then_with(|| a.family.cmp(&b.family))
    });
    top_families.truncate(top);

    // Top exact stacks.
    samples.sort_by(|a, b| match b.tbk.cmp(&a.tbk) {
        Ordering::Equal => b.tb.cmp(&a.tb),
        ord => ord,
    });
    samples.truncate(top);

    Ok(ProfileSummary {
        command,
        total_blocks,
        total_bytes,
        total_samples,
        top_families,
        top_stacks: samples,
    })
}

struct FamilyAgg {
    tbk: i64,
    tb: i64,
    samples: usize,
    exemplar: Vec<String>,
    best_tbk: i64,
}

fn trim_stack(frames: &[String], limit: usize, focus_symbol: Option<&str>) -> Vec<String> {
    let symbols: Vec<String> = frames.iter().map(|f| frame_symbol(f).to_string()).collect();
    let mut first_relevant = 0;
    let mut found = false;
    if let Some(focus) = focus_symbol {
        for (i, s) in symbols.iter().enumerate() {
            if s == focus {
                first_relevant = i;
                found = true;
                break;
            }
        }
    }
    if !found {
        for (i, s) in symbols.iter().enumerate() {
            if !is_runtime_symbol(s) {
                first_relevant = i;
                break;
            }
        }
    }
    let trimmed: Vec<String> = symbols.into_iter().skip(first_relevant).collect();
    if limit == 0 || trimmed.len() <= limit {
        trimmed
    } else {
        trimmed.into_iter().take(limit).collect()
    }
}

fn format_int_with_commas(n: i64) -> String {
    let s = n.to_string();
    let negative = s.starts_with('-');
    let digits: &str = if negative { &s[1..] } else { &s };
    let bytes = digits.as_bytes();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, ch) in bytes.iter().enumerate() {
        let rem = bytes.len() - i;
        out.push(*ch as char);
        if rem > 1 && (rem - 1) % 3 == 0 {
            out.push(',');
        }
    }
    if negative {
        format!("-{out}")
    } else {
        out
    }
}

fn format_usize_with_commas(n: usize) -> String {
    format_int_with_commas(n as i64)
}

fn render_summary(summary: &ProfileSummary, stack_limit: usize) -> String {
    let mut lines = Vec::<String>::new();
    lines.push(format!("Command: {}", summary.command));
    lines.push(format!(
        "Totals: {} blocks, {} bytes, {} program points",
        format_int_with_commas(summary.total_blocks),
        format_int_with_commas(summary.total_bytes),
        format_usize_with_commas(summary.total_samples)
    ));
    lines.push(String::new());
    lines.push("Top families by allocation blocks:".to_string());
    for (idx, family) in summary.top_families.iter().enumerate() {
        let stack = trim_stack(&family.exemplar, stack_limit, Some(&family.family));
        lines.push(format!(
            "{}. {} | blocks={} bytes={} samples={}",
            idx + 1,
            family.family,
            format_int_with_commas(family.tbk),
            format_int_with_commas(family.tb),
            family.samples
        ));
        if !stack.is_empty() {
            lines.push(format!("   stack: {}", stack.join(" -> ")));
        }
    }
    lines.push(String::new());
    lines.push("Top exact stacks by allocation blocks:".to_string());
    for (idx, stack) in summary.top_stacks.iter().enumerate() {
        let trimmed = trim_stack(&stack.frames, stack_limit, Some(&stack.family));
        lines.push(format!(
            "{}. blocks={} bytes={} family={}",
            idx + 1,
            format_int_with_commas(stack.tbk),
            format_int_with_commas(stack.tb),
            stack.family
        ));
        if !trimmed.is_empty() {
            lines.push(format!("   stack: {}", trimmed.join(" -> ")));
        }
    }
    lines.join("\n")
}

// Used in `Cli::stack_limit` argument parsing — clamped to >= 0 by type.
#[allow(dead_code)]
fn validate(_: &Cli) {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn frame_symbol_strips_address_and_source_location() {
        let frame = "0x1015c86d0: tla_check::foo::bar (src/foo.rs:12:34)";
        assert_eq!(
            normalize_frame(frame),
            "tla_check::foo::bar (src/foo.rs:12:34)"
        );
        assert_eq!(frame_symbol(frame), "tla_check::foo::bar");
    }

    #[test]
    fn classify_family_prefers_workspace_frame() {
        let frames = vec![
            "0x1: <alloc::alloc::Global as core::alloc::Allocator>::allocate (alloc/src/alloc.rs:1:1)".to_string(),
            "0x2: alloc::sync::Arc<T>::new (alloc/src/sync.rs:1:1)".to_string(),
            "0x3: tla_check::binding_chain::BindingChain::cons (src/binding_chain.rs:1:1)".to_string(),
            "0x4: std::thread::spawn (std/src/thread/mod.rs:1:1)".to_string(),
        ];
        assert_eq!(
            classify_family(&frames),
            "tla_check::binding_chain::BindingChain::cons"
        );
    }

    fn sample_profile() -> Value {
        json!({
            "cmd": "target/worker_1/profiling/ty check ...",
            "ftbl": [
                "[root]",
                "0x1: alloc::sync::Arc<T>::new (alloc/src/sync.rs:1:1)",
                "0x2: tla_check::binding_chain::BindingChain::cons (src/binding_chain.rs:10:20)",
                "0x3: tla_value::value::except::apply (src/value.rs:20:30)",
                "0x4: hashbrown::raw::RawTable<T>::reserve (hashbrown/src/raw/mod.rs:1:1)"
            ],
            "pps": [
                {"tbk": 10, "tb": 1000, "fs": [1, 2]},
                {"tbk": 6, "tb": 600, "fs": [1, 2]},
                {"tbk": 9, "tb": 900, "fs": [1, 3]},
                {"tbk": 4, "tb": 400, "fs": [1, 4]}
            ]
        })
    }

    #[test]
    fn summarize_profile_groups_by_family() {
        let summary = summarize_profile(&sample_profile(), 3).unwrap();
        assert_eq!(summary.total_blocks, 29);
        assert_eq!(summary.total_bytes, 2900);
        assert_eq!(summary.total_samples, 4);
        assert_eq!(
            summary.top_families[0].family,
            "tla_check::binding_chain::BindingChain::cons"
        );
        assert_eq!(summary.top_families[0].tbk, 16);
        assert_eq!(
            summary.top_families[1].family,
            "tla_value::value::except::apply"
        );
        assert_eq!(summary.top_families[1].tbk, 9);
        assert_eq!(
            summary.top_families[2].family,
            "hashbrown::raw::RawTable<T>::reserve"
        );
        assert_eq!(summary.top_families[2].tbk, 4);
    }

    #[test]
    fn render_summary_includes_focused_stack() {
        let data = json!({
            "cmd": "target/worker_1/profiling/ty check ...",
            "ftbl": [
                "[root]",
                "0x1: alloc::sync::Arc<T>::new (alloc/src/sync.rs:1:1)",
                "0x2: tla_check::binding_chain::BindingChain::cons (src/binding_chain.rs:10:20)",
                "0x3: tla_check::compiled_guard::quantifier (src/guard.rs:20:30)",
                "0x4: tla_check::checker::run (src/checker.rs:30:40)"
            ],
            "pps": [
                {"tbk": 10, "tb": 1000, "fs": [1, 2, 3, 4]}
            ]
        });
        let summary = summarize_profile(&data, 1).unwrap();
        let rendered = render_summary(&summary, 2);
        assert!(rendered.contains("Top families by allocation blocks:"));
        assert!(rendered.contains("tla_check::binding_chain::BindingChain::cons"));
        assert!(rendered.contains(
            "stack: tla_check::binding_chain::BindingChain::cons -> tla_check::compiled_guard::quantifier"
        ));
    }

    #[test]
    fn format_commas() {
        assert_eq!(format_int_with_commas(0), "0");
        assert_eq!(format_int_with_commas(123), "123");
        assert_eq!(format_int_with_commas(1234), "1,234");
        assert_eq!(format_int_with_commas(1234567), "1,234,567");
        assert_eq!(format_int_with_commas(-1234567), "-1,234,567");
    }
}
