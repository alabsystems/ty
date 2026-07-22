// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty-mccctl fetch` — download an MCC benchmark input set and its consensus
//! answer key for local sweeps/correctness gates.
//!
//! Folds the former `scripts/mcc_fetch.sh` into the unified CLI, and adds:
//!   * support for BOTH the 2024 `raw-result-analysis.csv.zip` and the 2025
//!     `raw-result-analysis.csv.tar.gz` answer-key layouts (auto-detected),
//!   * automatic extraction of the per-model `*.tgz` archives inside
//!     `INPUTS-YEAR/`, so the result is directly runnable by `ty-mccctl sweep`
//!     / `history` / `scripts/mcc_oracle_eval.py`.
//!
//! Downloads/extraction shell out to `curl` and `tar` (as the contest VM and
//! the prior script do); no new crate dependencies.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use clap::Args;

/// Command-line arguments for the MCC corpus `fetch` command.
#[derive(Debug, Args)]
#[command(
    after_help = "Examples:\n  ty-mccctl fetch --year 2025\n  ty-mccctl fetch --year 2024 --root ~/mcc-benchmarks/2024\n  ty-mccctl fetch --year 2025 --answer-key-only\n\nLayout written under --root (default ~/mcc-benchmarks/YEAR):\n  archives/INPUTS-YEAR.tar.gz\n  inputs/INPUTS-YEAR/<Model>/{model.pnml,*.xml,...}   (per-model .tgz auto-extracted)\n  results/raw-result-analysis.csv                      (consensus answer key)"
)]
pub struct FetchArgs {
    /// MCC year to fetch.
    #[arg(long, default_value = "2025")]
    pub year: String,

    /// Destination root (default: ~/mcc-benchmarks/YEAR).
    #[arg(long, value_name = "DIR")]
    pub root: Option<PathBuf>,

    /// Archive base URL (default: https://mcc.lip6.fr/YEAR/archives).
    #[arg(long, value_name = "URL")]
    pub base_url: Option<String>,

    /// Fetch/extract the consensus answer key only (skip the model inputs).
    #[arg(long)]
    pub answer_key_only: bool,

    /// Download INPUTS-YEAR.tar.gz but do not extract it.
    #[arg(long)]
    pub no_extract_inputs: bool,

    /// Extract the outer archive but do not unpack the per-model `*.tgz`.
    #[arg(long)]
    pub no_extract_models: bool,

    /// Redownload existing archives and re-extract.
    #[arg(long)]
    pub force: bool,
}

fn home() -> Result<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .context("HOME is not set; pass --root explicitly")
}

fn run_cmd(cmd: &mut Command) -> Result<()> {
    let label = format!("{cmd:?}");
    let status = cmd
        .status()
        .with_context(|| format!("failed to spawn: {label}"))?;
    if !status.success() {
        bail!("command failed ({status}): {label}");
    }
    Ok(())
}

/// `curl -fL` the URL to `dest`, skipping when it already exists (unless force).
fn download(url: &str, dest: &Path, force: bool) -> Result<bool> {
    if !force {
        if let Ok(meta) = std::fs::metadata(dest) {
            if meta.len() > 0 {
                println!("skip existing: {}", dest.display());
                return Ok(true);
            }
        }
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create dir {}", parent.display()))?;
    }
    let tmp = dest.with_extension("tmp.part");
    let _ = std::fs::remove_file(&tmp);
    println!("download: {url}");
    let ok = run_cmd(
        Command::new("curl")
            .args(["-fL", "--retry", "3", "--retry-delay", "2", "-o"])
            .arg(&tmp)
            .arg(url),
    );
    match ok {
        Ok(()) => {
            std::fs::rename(&tmp, dest)
                .with_context(|| format!("rename {} -> {}", tmp.display(), dest.display()))?;
            Ok(true)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

fn extract_tar_gz(archive: &Path, dest_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dest_dir)
        .with_context(|| format!("create dir {}", dest_dir.display()))?;
    println!("extract: {} -> {}", archive.display(), dest_dir.display());
    run_cmd(
        Command::new("tar")
            .arg("-xzf")
            .arg(archive)
            .arg("-C")
            .arg(dest_dir),
    )
}

/// Unpack every per-model `*.tgz` inside `inputs_root` into a sibling directory.
fn extract_per_model(inputs_root: &Path) -> Result<usize> {
    let mut n = 0usize;
    let entries = std::fs::read_dir(inputs_root)
        .with_context(|| format!("read dir {}", inputs_root.display()))?;
    for entry in entries {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("tgz") {
            run_cmd(
                Command::new("tar")
                    .arg("-xzf")
                    .arg(&path)
                    .arg("-C")
                    .arg(inputs_root),
            )?;
            n += 1;
        }
    }
    Ok(n)
}

/// Fetches and extracts the MCC benchmark corpus for the requested year under
/// `--root`.
///
/// # Errors
///
/// Returns an error if the home directory cannot be resolved, a download or
/// archive extraction fails, or the corpus layout cannot be written.
pub fn run(args: FetchArgs) -> Result<()> {
    let root = match args.root {
        Some(r) => r,
        None => home()?.join("mcc-benchmarks").join(&args.year),
    };
    let base = args
        .base_url
        .unwrap_or_else(|| format!("https://mcc.lip6.fr/{}/archives", args.year));
    let archives = root.join("archives");
    let results_dir = root.join("results");

    // --- Answer key (consensus oracle). Try .tar.gz (2025+) then .zip (2024). ---
    std::fs::create_dir_all(&results_dir)?;
    let key_csv = results_dir.join("raw-result-analysis.csv");
    if args.force
        || std::fs::metadata(&key_csv)
            .map(|m| m.len() == 0)
            .unwrap_or(true)
    {
        let tgz = archives.join("raw-result-analysis.csv.tar.gz");
        let zip = archives.join("raw-result-analysis.csv.zip");
        if download(
            &format!("{base}/raw-result-analysis.csv.tar.gz"),
            &tgz,
            args.force,
        )
        .is_ok()
        {
            extract_tar_gz(&tgz, &results_dir)?;
        } else if download(
            &format!("{base}/raw-result-analysis.csv.zip"),
            &zip,
            args.force,
        )
        .is_ok()
        {
            println!("extract: {} -> {}", zip.display(), results_dir.display());
            run_cmd(
                Command::new("unzip")
                    .arg("-o")
                    .arg(&zip)
                    .arg("-d")
                    .arg(&results_dir),
            )?;
        } else {
            bail!("could not fetch the answer key (.tar.gz or .zip) from {base}");
        }
    } else {
        println!("skip existing: {}", key_csv.display());
    }

    // --- Model inputs ---
    if !args.answer_key_only {
        let inputs_archive = archives.join(format!("INPUTS-{}.tar.gz", args.year));
        download(
            &format!("{base}/INPUTS-{}.tar.gz", args.year),
            &inputs_archive,
            args.force,
        )?;
        if !args.no_extract_inputs {
            let inputs_dir = root.join("inputs");
            extract_tar_gz(&inputs_archive, &inputs_dir)?;
            let inputs_root = inputs_dir.join(format!("INPUTS-{}", args.year));
            if !args.no_extract_models && inputs_root.is_dir() {
                let n = extract_per_model(&inputs_root)?;
                println!(
                    "unpacked {n} per-model archives in {}",
                    inputs_root.display()
                );
            }
        }
    }

    println!("\nMCC {} fetch complete.", args.year);
    println!("  root:       {}", root.display());
    if !args.answer_key_only {
        println!(
            "  inputs:     {}",
            root.join("inputs")
                .join(format!("INPUTS-{}", args.year))
                .display()
        );
    }
    println!("  answer key: {}", key_csv.display());
    Ok(())
}
