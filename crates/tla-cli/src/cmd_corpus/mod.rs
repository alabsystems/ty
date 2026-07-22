// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty corpus` subcommand: download and cache the TLA+ benchmark corpus used by
//! the TY-vs-TLC comparison harnesses and the `spec_regression` test.
//!
//! The corpus is the `specifications/` tree of github.com/tlaplus/Examples,
//! pinned to the commit recorded in `tests/tlc_comparison/spec_baseline.json`
//! (`05c7256`). It is NOT checked into this repo; many scripts and tests expect
//! it at `~/tlaplus-examples` (override with `TLAPLUS_EXAMPLES`). Two fetch
//! modes:
//!   * default: download the reproducible release asset published on
//!     `github.com/alabsystems/ty` (offline-friendly, sha256-verified);
//!   * `--from-upstream`: `git clone` tlaplus/Examples and checkout the pin.
//!
//! Follows the repo convention of shelling out to `curl`/`tar`/`git` rather
//! than adding an HTTP client dependency (cf. `scripts/mcc_fetch.sh`,
//! `scripts/fetch_tla_library.sh`).

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};

use crate::cli_schema::CorpusAction;

// `ty corpus sweep` (phase R0, docs/north-star-roadmap.md): the certify census
// over every `.cfg` in the corpus, as a self-service subcommand.
pub(crate) mod sweep;

/// The release tag + asset published by `gh release create corpus-v1-05c7256`.
const CORPUS_TAG: &str = "corpus-v1-05c7256";
const CORPUS_ASSET: &str = "ty-corpus-examples-05c7256.tar.gz";
const CORPUS_URL: &str = "https://github.com/alabsystems/ty/releases/download/corpus-v1-05c7256/ty-corpus-examples-05c7256.tar.gz";
/// sha256 of the published tarball (see the `.sha256` sidecar asset).
const CORPUS_SHA256: &str = "caff7e8c5be876f574ced9630b396f4dd039354063856ba6346e4b16ff0498b7";
/// The upstream pin (matches `tests/tlc_comparison/spec_baseline.json`).
const UPSTREAM_REPO: &str = "https://github.com/tlaplus/Examples";
const UPSTREAM_PIN: &str = "05c7256f153cd18e40e27eaa6dd4e60f4fcba15b";

/// Dispatch the `ty corpus <action>` subcommand.
pub(crate) fn cmd_corpus(action: CorpusAction) -> Result<()> {
    match action {
        CorpusAction::Fetch {
            dest,
            from_upstream,
            force,
        } => do_fetch(dest, from_upstream, force),
        CorpusAction::Path { dest } => {
            println!("{}", resolve_dest(dest).display());
            Ok(())
        }
        CorpusAction::Verify { dest } => do_verify(resolve_dest(dest)),
        CorpusAction::Sweep {
            dest,
            timeout,
            jobs,
            filter,
            format,
            out,
        } => sweep::cmd_sweep(dest, timeout, jobs, filter, format, out.as_deref()),
    }
}

/// Resolve the corpus install directory, mirroring every consumer's contract:
/// 1. explicit `--dest`, else 2. `$TLAPLUS_EXAMPLES`, else 3. `$HOME/tlaplus-examples`.
///
/// The corpus is laid out so that `<dest>/specifications/<dir>/<spec>.tla` resolves.
fn resolve_dest(dest: Option<PathBuf>) -> PathBuf {
    if let Some(d) = dest {
        return d;
    }
    if let Ok(env) = std::env::var("TLAPLUS_EXAMPLES") {
        if !env.is_empty() {
            // TLAPLUS_EXAMPLES historically points AT the `specifications` dir in
            // some consumers and at its parent in others; normalize to the parent
            // (the dir that CONTAINS `specifications`) for the install root.
            let p = PathBuf::from(&env);
            if p.file_name().and_then(|s| s.to_str()) == Some("specifications") {
                if let Some(parent) = p.parent() {
                    return parent.to_path_buf();
                }
            }
            return p;
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join("tlaplus-examples")
}

fn specifications_dir(dest: &Path) -> PathBuf {
    dest.join("specifications")
}

/// Probe whether a non-empty corpus is already installed at the default dir.
/// Returns the resolved install root and the count of `.cfg` specs found.
/// Used by `ty supremacy reproduce` to decide whether to auto-fetch.
pub(crate) fn probe_default() -> (PathBuf, usize) {
    let dest = resolve_dest(None);
    let n = count_cfgs(&specifications_dir(&dest));
    (dest, n)
}

fn do_fetch(dest: Option<PathBuf>, from_upstream: bool, force: bool) -> Result<()> {
    let dest = resolve_dest(dest);
    let specs = specifications_dir(&dest);

    if specs.is_dir() && !force {
        let n = count_cfgs(&specs);
        println!(
            "corpus already present at {} ({} .cfg specs); use --force to re-fetch",
            dest.display(),
            n
        );
        return Ok(());
    }

    std::fs::create_dir_all(&dest)
        .with_context(|| format!("creating corpus dir {}", dest.display()))?;

    if from_upstream {
        fetch_from_upstream(&dest)?;
    } else {
        fetch_from_release(&dest)?;
    }

    let n = count_cfgs(&specifications_dir(&dest));
    if n == 0 {
        bail!(
            "fetch completed but no .cfg specs found under {} — corpus layout unexpected",
            specifications_dir(&dest).display()
        );
    }
    println!(
        "corpus ready at {} ({} .cfg specs). Set TLAPLUS_EXAMPLES={} to point tools here.",
        dest.display(),
        n,
        dest.display()
    );
    Ok(())
}

/// Release-asset mode (default): download the published tarball, verify its
/// sha256, and extract so `<dest>/specifications/...` resolves.
///
/// We download via `gh release download` (auth-aware, so it also works when
/// the release asset is not reachable anonymously) when `gh` is available,
/// and fall back to `curl` of the asset URL if `gh` is absent. The sha256 is
/// verified either way.
fn fetch_from_release(dest: &Path) -> Result<()> {
    let dl_dir = std::env::temp_dir().join("ty-corpus-dl");
    let _ = std::fs::remove_dir_all(&dl_dir);
    std::fs::create_dir_all(&dl_dir).with_context(|| format!("creating {}", dl_dir.display()))?;
    let tmp = dl_dir.join(CORPUS_ASSET);

    let gh_ok = Command::new("gh")
        .args(["--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if gh_ok {
        eprintln!("downloading {CORPUS_ASSET} via gh (release {CORPUS_TAG})");
        run(
            Command::new("gh")
                .args([
                    "release",
                    "download",
                    CORPUS_TAG,
                    "--repo",
                    "alabsystems/ty",
                    "-p",
                    CORPUS_ASSET,
                    "-D",
                ])
                .arg(&dl_dir)
                .arg("--clobber"),
            "gh release download of corpus asset",
        )?;
    } else {
        eprintln!("gh not found; downloading {CORPUS_URL} via curl");
        run(
            Command::new("curl")
                .args(["-fsSL", "--retry", "3", "-o"])
                .arg(&tmp)
                .arg(CORPUS_URL),
            "curl download of corpus asset",
        )?;
    }

    verify_sha256(&tmp, CORPUS_SHA256)
        .context("corpus tarball sha256 mismatch — refusing to extract a corrupt/altered asset")?;

    eprintln!("extracting into {}", dest.display());
    run(
        Command::new("tar")
            .arg("-xzf")
            .arg(&tmp)
            .arg("-C")
            .arg(dest),
        "tar extract of corpus asset",
    )?;
    let _ = std::fs::remove_dir_all(&dl_dir);
    Ok(())
}

/// Upstream mode: clone tlaplus/Examples and checkout the recorded pin so the
/// corpus matches the TLC baselines in `spec_baseline.json` exactly.
fn fetch_from_upstream(dest: &Path) -> Result<()> {
    let work = std::env::temp_dir().join("ty-corpus-upstream");
    let _ = std::fs::remove_dir_all(&work);
    eprintln!("cloning {UPSTREAM_REPO} @ {UPSTREAM_PIN}");
    run(
        Command::new("git")
            .arg("clone")
            .arg(UPSTREAM_REPO)
            .arg(&work),
        "git clone tlaplus/Examples",
    )?;
    run(
        Command::new("git")
            .arg("-C")
            .arg(&work)
            .args(["checkout", UPSTREAM_PIN]),
        "git checkout corpus pin",
    )?;
    let src = work.join("specifications");
    if !src.is_dir() {
        bail!(
            "upstream clone missing specifications/ at {}",
            src.display()
        );
    }
    // Copy specifications/ into dest (cp -R; portable enough for this tool).
    run(
        Command::new("cp").arg("-R").arg(&src).arg(dest),
        "copy specifications/ into corpus dir",
    )?;
    let _ = std::fs::remove_dir_all(&work);
    Ok(())
}

fn do_verify(dest: PathBuf) -> Result<()> {
    let specs = specifications_dir(&dest);
    if !specs.is_dir() {
        bail!(
            "corpus NOT found: {} does not exist. Run `ty corpus fetch`.",
            specs.display()
        );
    }
    let n = count_cfgs(&specs);
    if n == 0 {
        bail!(
            "corpus present but empty ({} has no .cfg specs). Re-run `ty corpus fetch --force`.",
            specs.display()
        );
    }
    println!("corpus OK at {} ({} .cfg specs)", dest.display(), n);
    Ok(())
}

/// Count `.cfg` files under a directory (recursive), the cheap presence metric.
fn count_cfgs(dir: &Path) -> usize {
    fn walk(dir: &Path, acc: &mut usize) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, acc);
            } else if p.extension().and_then(|s| s.to_str()) == Some("cfg") {
                *acc += 1;
            }
        }
    }
    let mut acc = 0;
    walk(dir, &mut acc);
    acc
}

/// Verify a file's sha256 against `expected` (hex). Shells out to `shasum`/
/// `sha256sum` (whichever exists) per the repo's no-extra-dep convention.
fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    let out = sha256_hex(path)?;
    if !out.eq_ignore_ascii_case(expected) {
        bail!("sha256 mismatch: got {out}, expected {expected}");
    }
    Ok(())
}

fn sha256_hex(path: &Path) -> Result<String> {
    // Try `shasum -a 256` (macOS/BSD/perl) then `sha256sum` (GNU/Linux).
    for (bin, args) in [("shasum", vec!["-a", "256"]), ("sha256sum", vec![])] {
        let mut cmd = Command::new(bin);
        cmd.args(&args).arg(path);
        if let Ok(out) = cmd.output() {
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout);
                if let Some(hex) = s.split_whitespace().next() {
                    return Ok(hex.to_string());
                }
            }
        }
    }
    bail!("no usable sha256 tool found (need `shasum` or `sha256sum` on PATH)")
}

fn run(cmd: &mut Command, what: &str) -> Result<()> {
    let status = cmd
        .status()
        .with_context(|| format!("failed to spawn for {what}"))?;
    if !status.success() {
        bail!("{what} failed with status {status}");
    }
    Ok(())
}

/// The corpus identity the sweep header reports: (release tag, upstream pin).
fn corpus_identity() -> (&'static str, &'static str) {
    (CORPUS_TAG, UPSTREAM_PIN)
}
