// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! TLC↔TY state-graph differential.
//!
//! Ported from `scripts/compare_tlc_ty.py` so MCC-adjacent tooling has
//! a single compiler-enforced interface. Compares a TLC DOT dump
//! against a TY successor-debug transcript for state-set and
//! transition parity.
//!
//! Two artefacts are required, plus optional provenance sidecars:
//!
//! 1. TLC DOT: `java -jar tytools.jar -dump dot,actionlabels …`.
//! 2. TY debug: `TY_DEBUG_SUCCESSORS=1 TY_DEBUG_SUCCESSORS_TLCFP=1
//!    cargo run --bin ty -- check … 2>&1 > out.txt`.
//!
//! Provenance sidecars (`<artifact>.provenance.json`) pin the
//! `(spec_path, cfg_path, cfg_sha256)` triple for both artefacts so
//! identity drift fails fast — the qualification-1 lesson applied to
//! the comparison harness.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use tla_petri::tlc_dot::{parse_tlc_dot, TlcStateGraph, TlcTransition};

#[derive(Parser, Debug)]
#[command(
    name = "ty-tlc-compare",
    about = "Compare a TLC DOT state graph against TY successor-debug output",
    long_about = "Differential check between a TLC `-dump dot,actionlabels` artefact and a \
                  TY successor-debug transcript. Catches state-set and transition \
                  divergence and enforces matching provenance sidecars to prevent \
                  the cross-tool config-drift class of bugs."
)]
struct Cli {
    /// TLC DOT file produced via `-dump dot,actionlabels`.
    #[arg(value_name = "TLC_DOT")]
    tlc_dot: PathBuf,

    /// TY successor-debug output file.
    #[arg(value_name = "TY_DEBUG")]
    ty_debug: PathBuf,

    /// Print extra divergence detail.
    #[arg(short, long)]
    verbose: bool,

    /// Provenance sidecar path for the TLC artefact (default: alongside).
    #[arg(long, value_name = "PATH")]
    tlc_provenance: Option<PathBuf>,

    /// Provenance sidecar path for the TY artefact (default: alongside).
    #[arg(long, value_name = "PATH")]
    ty_provenance: Option<PathBuf>,

    /// Skip provenance enforcement (downgrade to warning).
    #[arg(long)]
    allow_missing_provenance: bool,

    /// Write provenance sidecars for both artefacts before comparing.
    #[arg(long)]
    write_provenance: bool,

    /// Spec path (required with `--write-provenance`).
    #[arg(long, value_name = "PATH")]
    spec_path: Option<String>,

    /// Config path (required with `--write-provenance`).
    #[arg(long, value_name = "PATH")]
    cfg_path: Option<String>,

    /// Config SHA-256; otherwise computed from `--cfg-path`.
    #[arg(long, value_name = "HEX")]
    cfg_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TyState {
    fingerprint: i64,
    depth: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TySuccessor {
    src_fp: i64,
    dst_fp: i64,
    #[allow(dead_code)]
    action: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct TyDebugOutput {
    states: BTreeMap<i64, TyState>,
    successors: Vec<TySuccessor>,
    initial_states: BTreeSet<i64>,
    depth_groups: BTreeMap<usize, BTreeSet<i64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct ConfigProvenanceSource {
    spec_path: String,
    cfg_path: String,
    cfg_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ConfigProvenanceDoc {
    schema_version: u32,
    source: ConfigProvenanceSource,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("{err:#}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: &Cli) -> Result<ExitCode> {
    if !cli.tlc_dot.exists() {
        bail!("Error: TLC DOT file not found: {}", cli.tlc_dot.display());
    }
    if !cli.ty_debug.exists() {
        bail!("Error: TY debug file not found: {}", cli.ty_debug.display());
    }

    let tlc_prov_path = cli
        .tlc_provenance
        .clone()
        .unwrap_or_else(|| default_provenance_path(&cli.tlc_dot));
    let ty_prov_path = cli
        .ty_provenance
        .clone()
        .unwrap_or_else(|| default_provenance_path(&cli.ty_debug));

    if cli.write_provenance {
        let spec_path = cli.spec_path.as_deref().ok_or_else(|| {
            anyhow!("Error: --write-provenance requires --spec-path and --cfg-path")
        })?;
        let cfg_path = cli.cfg_path.as_deref().ok_or_else(|| {
            anyhow!("Error: --write-provenance requires --spec-path and --cfg-path")
        })?;
        let cfg_sha256 = match cli.cfg_sha256.as_deref() {
            Some(value) => value.to_string(),
            None => read_cfg_sha256(cfg_path)?,
        };
        let provenance = ConfigProvenanceSource {
            spec_path: spec_path.to_string(),
            cfg_path: cfg_path.to_string(),
            cfg_sha256,
        };
        write_config_provenance(&tlc_prov_path, &provenance)?;
        write_config_provenance(&ty_prov_path, &provenance)?;
    }

    match (
        load_config_provenance(&tlc_prov_path, "TLC"),
        load_config_provenance(&ty_prov_path, "TY"),
    ) {
        (Ok(tlc_prov), Ok(ty_prov)) => {
            assert_same_config_provenance(&tlc_prov, &ty_prov)?;
        }
        (Err(err), _) | (_, Err(err)) => {
            if cli.allow_missing_provenance && err.to_string().contains("not found") {
                eprintln!("Warning: {err}");
                eprintln!("Warning: skipping provenance identity check");
            } else {
                return Err(anyhow!("Error: {err}"));
            }
        }
    }

    println!("Loading TLC DOT: {}", cli.tlc_dot.display());
    let tlc_text = fs::read_to_string(&cli.tlc_dot)
        .with_context(|| format!("reading TLC DOT {}", cli.tlc_dot.display()))?;
    let tlc_graph = parse_tlc_dot(&tlc_text).map_err(|e| anyhow!(e))?;
    println!(
        "  Loaded {} states, {} transitions",
        format_with_commas(tlc_graph.states.len() as u64),
        format_with_commas(tlc_graph.transitions.len() as u64)
    );
    println!();

    println!("Loading TY debug: {}", cli.ty_debug.display());
    let ty_text = fs::read_to_string(&cli.ty_debug)
        .with_context(|| format!("reading TY debug {}", cli.ty_debug.display()))?;
    let ty_output = parse_ty_debug(&ty_text);
    println!(
        "  Loaded {} states, {} transitions",
        format_with_commas(ty_output.states.len() as u64),
        format_with_commas(ty_output.successors.len() as u64)
    );
    println!();

    let code = compare_graphs(&tlc_graph, &ty_output, cli.verbose);
    Ok(ExitCode::from(code))
}

fn default_provenance_path(artifact: &Path) -> PathBuf {
    let mut new_name = artifact
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    new_name.push(".provenance.json");
    artifact.with_file_name(new_name)
}

fn read_cfg_sha256(cfg_path: &str) -> Result<String> {
    let path = Path::new(cfg_path);
    if !path.exists() {
        bail!(
            "Cannot compute cfg_sha256: cfg file does not exist: {}",
            path.display()
        );
    }
    let bytes = fs::read(path).with_context(|| format!("reading cfg file {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn write_config_provenance(path: &Path, provenance: &ConfigProvenanceSource) -> Result<()> {
    let doc = ConfigProvenanceDoc {
        schema_version: 1,
        source: provenance.clone(),
    };
    let serialised = serde_json::to_string_pretty(&doc).context("serialising provenance JSON")?;
    let mut output = serialised;
    output.push('\n');
    fs::write(path, output)
        .with_context(|| format!("writing provenance sidecar {}", path.display()))?;
    Ok(())
}

fn load_config_provenance(path: &Path, kind: &str) -> Result<ConfigProvenanceSource> {
    if !path.exists() {
        bail!(
            "{kind} provenance sidecar not found: {}\nCreate it with --write-provenance --spec-path ... --cfg-path ...",
            path.display()
        );
    }
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading {kind} provenance {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
        anyhow!(
            "{kind} provenance sidecar is not valid JSON: {}: {e}",
            path.display()
        )
    })?;
    let source_value = value
        .get("source")
        .cloned()
        .unwrap_or_else(|| value.clone());
    let spec_path = source_value
        .get("spec_path")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let cfg_path = source_value
        .get("cfg_path")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let cfg_sha256 = source_value
        .get("cfg_sha256")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let mut missing: Vec<&str> = Vec::new();
    if spec_path.as_deref().map(str::is_empty).unwrap_or(true) {
        missing.push("spec_path");
    }
    if cfg_path.as_deref().map(str::is_empty).unwrap_or(true) {
        missing.push("cfg_path");
    }
    if cfg_sha256.as_deref().map(str::is_empty).unwrap_or(true) {
        missing.push("cfg_sha256");
    }
    if !missing.is_empty() {
        bail!(
            "{kind} provenance missing required fields {missing:?}: {}",
            path.display()
        );
    }
    Ok(ConfigProvenanceSource {
        spec_path: spec_path.unwrap(),
        cfg_path: cfg_path.unwrap(),
        cfg_sha256: cfg_sha256.unwrap(),
    })
}

fn assert_same_config_provenance(
    tlc: &ConfigProvenanceSource,
    ty: &ConfigProvenanceSource,
) -> Result<()> {
    let mut mismatches: Vec<String> = Vec::new();
    if tlc.spec_path != ty.spec_path {
        mismatches.push(format!(
            "spec_path: TLC={:?} TY={:?}",
            tlc.spec_path, ty.spec_path
        ));
    }
    if tlc.cfg_path != ty.cfg_path {
        mismatches.push(format!(
            "cfg_path: TLC={:?} TY={:?}",
            tlc.cfg_path, ty.cfg_path
        ));
    }
    if tlc.cfg_sha256 != ty.cfg_sha256 {
        mismatches.push(format!(
            "cfg_sha256: TLC={:?} TY={:?}",
            tlc.cfg_sha256, ty.cfg_sha256
        ));
    }
    if !mismatches.is_empty() {
        bail!("Config provenance mismatch: {}", mismatches.join("; "));
    }
    Ok(())
}

fn parse_ty_debug(text: &str) -> TyDebugOutput {
    let mut out = TyDebugOutput::default();
    let mut current: Option<i64> = None;

    for line in text.lines() {
        if let Some((fp, depth)) = parse_state_line(line) {
            out.states.insert(
                fp,
                TyState {
                    fingerprint: fp,
                    depth,
                },
            );
            out.depth_groups.entry(depth).or_default().insert(fp);
            if depth == 0 {
                out.initial_states.insert(fp);
            }
            current = Some(fp);
            continue;
        }
        if let Some(fp) = parse_from_state_line(line) {
            current = Some(fp);
            continue;
        }
        if let Some((dst, action)) = parse_succ_line(line) {
            if let Some(src) = current {
                out.successors.push(TySuccessor {
                    src_fp: src,
                    dst_fp: dst,
                    action,
                });
            }
        }
    }

    out
}

/// `STATE <internal> tlc=<hex> depth=<int>` (anything after is ignored).
fn parse_state_line(line: &str) -> Option<(i64, usize)> {
    let trimmed = line.trim_start();
    let rest = trimmed.strip_prefix("STATE ")?;
    // skip internal fp token
    let (_internal, rest) = split_one_token(rest)?;
    let rest = rest.trim_start();
    let after_tlc = rest.strip_prefix("tlc=")?;
    let (tlc_token, rest) = split_one_token(after_tlc)?;
    let fp = hex_u64_to_signed_i64(tlc_token).ok()?;
    let rest = rest.trim_start();
    let after_depth = rest.strip_prefix("depth=")?;
    let (depth_token, _rest) = split_one_token(after_depth)?;
    let depth = depth_token.parse::<usize>().ok()?;
    Some((fp, depth))
}

/// `from_state <id> tlc=0x<hex>` — older debug shape.
fn parse_from_state_line(line: &str) -> Option<i64> {
    let trimmed = line.trim_start();
    let rest = trimmed.strip_prefix("from_state ")?;
    let (_id_token, rest) = split_one_token(rest)?;
    let rest = rest.trim_start();
    let after_tlc = rest.strip_prefix("tlc=")?;
    let (tlc_token, _rest) = split_one_token(after_tlc)?;
    let token = tlc_token.strip_prefix("0x").unwrap_or(tlc_token);
    hex_u64_to_signed_i64(token).ok()
}

/// Successor line.
///
/// Two shapes:
/// * `succ internal=<fp> tlc=<fp>[ changes=N][ action=Name]`
/// * `succ <id> tlc=0x<hex>[ (action=Name)]`
fn parse_succ_line(line: &str) -> Option<(i64, Option<String>)> {
    let trimmed = line.trim_start();
    let rest = trimmed.strip_prefix("succ ")?;

    // Detailed format: internal=<fp> tlc=<fp> ...
    if let Some(after_internal) = rest.strip_prefix("internal=") {
        let (_internal_tok, after) = split_one_token(after_internal)?;
        let after = after.trim_start();
        let after_tlc = after.strip_prefix("tlc=")?;
        let (tlc_token, tail) = split_one_token(after_tlc)?;
        let fp = hex_u64_to_signed_i64(tlc_token).ok()?;
        let action = extract_action_keyword(tail).or_else(|| extract_paren_action(tail));
        return Some((fp, action));
    }

    // Alternate: succ <id> tlc=0x<hex> [(action=Name)]
    let (_id_token, after) = split_one_token(rest)?;
    let after = after.trim_start();
    let after_tlc = after.strip_prefix("tlc=")?;
    let (tlc_token, tail) = split_one_token(after_tlc)?;
    let token = tlc_token.strip_prefix("0x").unwrap_or(tlc_token);
    let fp = hex_u64_to_signed_i64(token).ok()?;
    let action = extract_paren_action(tail).or_else(|| extract_action_keyword(tail));
    Some((fp, action))
}

fn extract_action_keyword(tail: &str) -> Option<String> {
    for token in tail.split_whitespace() {
        if let Some(value) = token.strip_prefix("action=") {
            return Some(value.trim_end_matches(')').to_string());
        }
    }
    None
}

fn extract_paren_action(tail: &str) -> Option<String> {
    let open = tail.find('(')?;
    let rest = &tail[open + 1..];
    let close = rest.find(')')?;
    let inside = &rest[..close];
    let value = inside.strip_prefix("action=")?;
    Some(value.to_string())
}

fn split_one_token(rest: &str) -> Option<(&str, &str)> {
    let rest = rest.trim_start_matches([' ', '\t']);
    if rest.is_empty() {
        return None;
    }
    let end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
    let (token, remainder) = rest.split_at(end);
    Some((token, remainder))
}

/// `int(hex, 16)` followed by two's-complement folding into i64.
fn hex_u64_to_signed_i64(hex_str: &str) -> Result<i64, std::num::ParseIntError> {
    let cleaned = hex_str.strip_prefix("0x").unwrap_or(hex_str);
    let value = u64::from_str_radix(cleaned, 16)?;
    Ok(value as i64)
}

fn compare_graphs(tlc: &TlcStateGraph, ty: &TyDebugOutput, verbose: bool) -> u8 {
    let tlc_fps: BTreeSet<i64> = tlc.states.keys().copied().collect();
    let ty_fps: BTreeSet<i64> = ty.states.keys().copied().collect();

    println!("State Graph Comparison");
    println!("{}", "=".repeat(40));
    println!();
    println!("TLC States:  {}", format_with_commas(tlc_fps.len() as u64));
    println!("TY States: {}", format_with_commas(ty_fps.len() as u64));
    println!();

    let common: BTreeSet<i64> = tlc_fps.intersection(&ty_fps).copied().collect();
    let tlc_only: BTreeSet<i64> = tlc_fps.difference(&ty_fps).copied().collect();
    let ty_only: BTreeSet<i64> = ty_fps.difference(&tlc_fps).copied().collect();

    println!("Fingerprint Analysis:");
    println!("  Common:    {}", format_with_commas(common.len() as u64));
    println!("  TLC-only:  {}", format_with_commas(tlc_only.len() as u64));
    println!("  TY-only: {}", format_with_commas(ty_only.len() as u64));
    println!();

    if tlc_only.is_empty() && ty_only.is_empty() {
        println!("MATCH: Both tools found identical state sets.");
        let tlc_edges: BTreeSet<(i64, i64)> = tlc
            .transitions
            .iter()
            .map(|t| (t.src_fp, t.dst_fp))
            .collect();
        let ty_edges: BTreeSet<(i64, i64)> =
            ty.successors.iter().map(|s| (s.src_fp, s.dst_fp)).collect();
        let common_edges = tlc_edges.intersection(&ty_edges).count();
        let tlc_only_edges = tlc_edges.difference(&ty_edges).count();
        let ty_only_edges = ty_edges.difference(&tlc_edges).count();
        println!();
        println!("Transition Analysis:");
        println!(
            "  TLC transitions:  {}",
            format_with_commas(tlc_edges.len() as u64)
        );
        println!(
            "  TY transitions: {}",
            format_with_commas(ty_edges.len() as u64)
        );
        println!(
            "  Common:           {}",
            format_with_commas(common_edges as u64)
        );
        println!(
            "  TLC-only:         {}",
            format_with_commas(tlc_only_edges as u64)
        );
        println!(
            "  TY-only:        {}",
            format_with_commas(ty_only_edges as u64)
        );
        if tlc_only_edges == 0 && ty_only_edges == 0 {
            println!();
            println!("PERFECT MATCH: State sets AND transitions are identical.");
            return 0;
        }
        println!();
        println!("WARNING: State sets match but transitions differ.");
        return 1;
    }

    println!("Per-Depth Analysis:");
    let max_depth = tlc
        .depth_groups
        .keys()
        .copied()
        .chain(ty.depth_groups.keys().copied())
        .max()
        .unwrap_or(0);

    let mut first_diverge: Option<usize> = None;
    for depth in 0..=max_depth {
        let empty = BTreeSet::new();
        let tlc_at = tlc.depth_groups.get(&depth).unwrap_or(&empty);
        let ty_at = ty.depth_groups.get(&depth).unwrap_or(&empty);
        let status = if tlc_at == ty_at { "MATCH" } else { "DIVERGE" };
        if status == "DIVERGE" && first_diverge.is_none() {
            first_diverge = Some(depth);
        }
        println!(
            "  Depth {depth}: TLC={}, TY={} ({})",
            format_with_commas(tlc_at.len() as u64),
            format_with_commas(ty_at.len() as u64),
            status
        );
    }
    println!();

    if let Some(depth) = first_diverge {
        println!("First Divergence: Depth {depth}");
        println!();
        let empty = BTreeSet::new();
        let tlc_at = tlc.depth_groups.get(&depth).unwrap_or(&empty);
        let ty_at = ty.depth_groups.get(&depth).unwrap_or(&empty);
        let missing_in_ty: Vec<i64> = tlc_at.difference(ty_at).copied().collect();
        let missing_in_tlc: Vec<i64> = ty_at.difference(tlc_at).copied().collect();

        if !missing_in_ty.is_empty() {
            println!("Example states in TLC but not TY:");
            for fp in missing_in_ty.iter().take(5) {
                if let Some(state) = tlc.states.get(fp) {
                    let label_preview: String = state.label.chars().take(80).collect();
                    println!("  FP: {fp}");
                    println!("      {label_preview}");
                    if let Some(pred) = find_predecessor(&tlc.transitions, *fp) {
                        let action_str = pred
                            .action
                            .as_ref()
                            .map(|a| format!(" via {a}"))
                            .unwrap_or_default();
                        println!("      Predecessor: {}{}", pred.src_fp, action_str);
                    }
                    println!();
                }
            }
        }
        if verbose && !missing_in_tlc.is_empty() {
            println!("Example states in TY but not TLC:");
            for fp in missing_in_tlc.iter().take(5) {
                println!("  FP: {fp}");
            }
            println!();
        }
    }

    1
}

fn find_predecessor(transitions: &[TlcTransition], dst: i64) -> Option<&TlcTransition> {
    transitions.iter().find(|t| t.dst_fp == dst)
}

fn format_with_commas(n: u64) -> String {
    let raw = n.to_string();
    let mut out = String::with_capacity(raw.len() + raw.len() / 3);
    for (i, c) in raw.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out.chars().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_minimal_artifacts(tmp: &Path) -> (PathBuf, PathBuf) {
        let tlc_dot = tmp.join("tlc.dot");
        let ty_debug = tmp.join("ty_debug.txt");
        fs::write(
            &tlc_dot,
            "digraph DiskGraph {\n  1 [label=\"/\\\\ x = 0\",style = filled];\n}\n",
        )
        .unwrap();
        fs::write(&ty_debug, "STATE deadbeef tlc=1 depth=0 -> 0 successors\n").unwrap();
        (tlc_dot, ty_debug)
    }

    fn write_provenance(path: &Path, spec_path: &str, cfg_path: &str, cfg_sha256: &str) {
        let doc = ConfigProvenanceDoc {
            schema_version: 1,
            source: ConfigProvenanceSource {
                spec_path: spec_path.into(),
                cfg_path: cfg_path.into(),
                cfg_sha256: cfg_sha256.into(),
            },
        };
        let text = serde_json::to_string_pretty(&doc).unwrap() + "\n";
        fs::write(path, text).unwrap();
    }

    #[test]
    fn missing_provenance_fails_by_default() {
        let tmp = tempfile::tempdir().unwrap();
        let (tlc_dot, ty_debug) = build_minimal_artifacts(tmp.path());
        let cli = Cli {
            tlc_dot,
            ty_debug,
            verbose: false,
            tlc_provenance: None,
            ty_provenance: None,
            allow_missing_provenance: false,
            write_provenance: false,
            spec_path: None,
            cfg_path: None,
            cfg_sha256: None,
        };
        let err = run(&cli).expect_err("default mode requires provenance");
        let msg = err.to_string();
        assert!(
            msg.contains("provenance sidecar not found"),
            "unexpected: {msg}"
        );
    }

    #[test]
    fn provenance_mismatch_fails_fast() {
        let tmp = tempfile::tempdir().unwrap();
        let (tlc_dot, ty_debug) = build_minimal_artifacts(tmp.path());
        let tlc_prov = default_provenance_path(&tlc_dot);
        let ty_prov = default_provenance_path(&ty_debug);
        write_provenance(
            &tlc_prov,
            "NanoBlockchain/MCNano.tla",
            "NanoBlockchain/MCNanoSmall.cfg",
            &"a".repeat(64),
        );
        write_provenance(
            &ty_prov,
            "NanoBlockchain/MCNano.tla",
            "/tmp/MCNanoSmall_no_view.cfg",
            &"b".repeat(64),
        );
        let cli = Cli {
            tlc_dot,
            ty_debug,
            verbose: false,
            tlc_provenance: None,
            ty_provenance: None,
            allow_missing_provenance: false,
            write_provenance: false,
            spec_path: None,
            cfg_path: None,
            cfg_sha256: None,
        };
        let err = run(&cli).expect_err("mismatched provenance should fail");
        let msg = err.to_string();
        assert!(
            msg.contains("Config provenance mismatch"),
            "unexpected: {msg}"
        );
    }

    #[test]
    fn matching_provenance_allows_compare() {
        let tmp = tempfile::tempdir().unwrap();
        let (tlc_dot, ty_debug) = build_minimal_artifacts(tmp.path());
        let tlc_prov = default_provenance_path(&tlc_dot);
        let ty_prov = default_provenance_path(&ty_debug);
        for path in [&tlc_prov, &ty_prov] {
            write_provenance(
                path,
                "NanoBlockchain/MCNano.tla",
                "NanoBlockchain/MCNanoSmall.cfg",
                &"a".repeat(64),
            );
        }
        let cli = Cli {
            tlc_dot,
            ty_debug,
            verbose: false,
            tlc_provenance: None,
            ty_provenance: None,
            allow_missing_provenance: false,
            write_provenance: false,
            spec_path: None,
            cfg_path: None,
            cfg_sha256: None,
        };
        let result = run(&cli).expect("provenance match allows compare");
        // Exit code 0 means PERFECT MATCH.
        assert_eq!(
            format!("{:?}", result),
            format!("{:?}", ExitCode::from(0)),
            "expected PERFECT MATCH (exit 0)"
        );
    }

    #[test]
    fn write_provenance_generates_sidecars() {
        let tmp = tempfile::tempdir().unwrap();
        let (tlc_dot, ty_debug) = build_minimal_artifacts(tmp.path());
        let cfg_path = tmp.path().join("MCNanoSmall.cfg");
        let cfg_contents = "SPECIFICATION Next\nVIEW View\n";
        fs::write(&cfg_path, cfg_contents).unwrap();
        let cli = Cli {
            tlc_dot: tlc_dot.clone(),
            ty_debug: ty_debug.clone(),
            verbose: false,
            tlc_provenance: None,
            ty_provenance: None,
            allow_missing_provenance: false,
            write_provenance: true,
            spec_path: Some("NanoBlockchain/MCNano.tla".to_string()),
            cfg_path: Some(cfg_path.to_string_lossy().to_string()),
            cfg_sha256: None,
        };
        run(&cli).expect("write provenance and compare");
        let mut hasher = Sha256::new();
        hasher.update(cfg_contents.as_bytes());
        let expected = format!("{:x}", hasher.finalize());

        for sidecar in [
            default_provenance_path(&tlc_dot),
            default_provenance_path(&ty_debug),
        ] {
            let payload: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(&sidecar).unwrap()).unwrap();
            let source = &payload["source"];
            assert_eq!(source["spec_path"], "NanoBlockchain/MCNano.tla");
            assert_eq!(
                source["cfg_path"].as_str().unwrap(),
                cfg_path.to_string_lossy().as_ref()
            );
            assert_eq!(source["cfg_sha256"], expected);
        }
    }

    #[test]
    fn parse_state_line_extracts_tlc_fp() {
        let parsed =
            parse_state_line("STATE 9dc3d08535a15415 tlc=0b9ed133720eec70 depth=0 -> 3 successors");
        assert!(parsed.is_some());
        let (fp, depth) = parsed.unwrap();
        assert_eq!(depth, 0);
        let expected = u64::from_str_radix("0b9ed133720eec70", 16).unwrap() as i64;
        assert_eq!(fp, expected);
    }

    #[test]
    fn parse_succ_line_detailed_format() {
        let parsed = parse_succ_line("  succ internal=abc tlc=1 changes=2 action=Move");
        assert!(parsed.is_some());
        let (fp, action) = parsed.unwrap();
        assert_eq!(fp, 1);
        assert_eq!(action.as_deref(), Some("Move"));
    }

    #[test]
    fn parse_succ_line_alternate_format() {
        let parsed = parse_succ_line("  succ 17 tlc=0x2a (action=Step)");
        assert!(parsed.is_some());
        let (fp, action) = parsed.unwrap();
        assert_eq!(fp, 0x2a);
        assert_eq!(action.as_deref(), Some("Step"));
    }

    #[test]
    fn hex_u64_folds_into_signed() {
        // 0xffffffffffffffff -> -1 in two's complement i64.
        assert_eq!(hex_u64_to_signed_i64("ffffffffffffffff").unwrap(), -1);
        assert_eq!(hex_u64_to_signed_i64("0").unwrap(), 0);
        assert_eq!(hex_u64_to_signed_i64("1").unwrap(), 1);
    }

    #[test]
    fn format_with_commas_groups_three() {
        assert_eq!(format_with_commas(0), "0");
        assert_eq!(format_with_commas(999), "999");
        assert_eq!(format_with_commas(1_234), "1,234");
        assert_eq!(format_with_commas(1_234_567), "1,234,567");
    }
}
