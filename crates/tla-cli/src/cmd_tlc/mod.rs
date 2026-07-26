// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty install-tlc` subcommand (alias: `ty tlc`): download and install TLC (the reference TLA+ model
//! checker) so the TLC-vs-TY comparison harnesses run with no manual setup.
//!
//! Installs two jars into `~/tlaplus/` (override with `--dest`):
//!   * `tytools.jar` — TLC itself. NOTE: the upstream STABLE release v1.7.4 is
//!     broken (its `tla2tools.jar` throws `NoClassDefFoundError` for
//!     `tlc2.value.impl.KSubsetValue`), so we pull the working `tla2tools.jar`
//!     from the *nightly* channel. Because nightly rolls we do NOT pin its
//!     sha256; instead we verify it is FUNCTIONAL (`tlc2.TLC -h` prints usage)
//!     and print the downloaded jar's sha256 for the user's record.
//!   * `CommunityModules.jar` — the community module library, pinned by sha256
//!     (an immutable release asset).
//!
//! Both land at the exact paths `ty supremacy compare` auto-discovers
//! (`default_tlc_jar()` / `default_community_modules_jar()` in
//! `cmd_supremacy/compare.rs`): `<dest>/tytools.jar` and
//! `<dest>/CommunityModules.jar`.
//!
//! A third, optional artifact completes the TLC-comparable toolchain:
//!   * `tla-library/` — the **upstream** TLA+ proof-system module library
//!     (`tlaplus/tlapm`'s `library/`, pinned per-file by sha256). 25 of the 141
//!     eligible corpus rows `EXTENDS TLAPS`/`FiniteSetTheorems`/
//!     `NaturalsInduction` and do not parse without it. Installed by
//!     `ty install-tlc proof-library` (or `install --with-proof-library`); see
//!     [`proof_library`].
//!
//! Follows the repo convention of shelling out to `curl`/`tar`/`java` rather
//! than adding an HTTP client dependency (cf. `cmd_corpus`).

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};

use crate::cli_schema::TlcAction;

pub(crate) mod proof_library;

/// The working TLC distribution. The stable release jar is broken, so we use
/// the nightly channel; this URL rolls, so it is NOT sha-pinned (we verify it
/// functionally instead and print the resulting sha256).
const TLA2TOOLS_NIGHTLY_URL: &str = "https://nightly.tlapl.us/dist/tla2tools.jar";

/// CommunityModules: an immutable, dated release asset — sha256-pinned.
const COMMUNITY_MODULES_URL: &str = "https://github.com/tlaplus/CommunityModules/releases/download/202604221529/CommunityModules-deps.jar";
/// sha256 of the pinned CommunityModules-deps.jar (verified on install).
const COMMUNITY_MODULES_SHA256: &str =
    "b074c4c507b84b4a92fafba1f543efe76a7bbef3a08877362c7f9c96d078b6a8";

/// Filenames at the install dir, matching `ty supremacy compare` discovery.
const TLC_JAR_NAME: &str = "tytools.jar";
const COMMUNITY_MODULES_NAME: &str = "CommunityModules.jar";

/// Dispatch the `ty install-tlc <action>` subcommand.
pub(crate) fn cmd_tlc(action: TlcAction) -> Result<()> {
    match action {
        TlcAction::Install {
            dest,
            force,
            with_proof_library,
        } => do_install(dest, force, with_proof_library),
        TlcAction::Path { dest } => {
            println!("{}", resolve_dest(dest).display());
            Ok(())
        }
        TlcAction::Verify { dest } => do_verify(resolve_dest(dest)),
        TlcAction::ProofLibrary { dest, force } => {
            proof_library::do_install(&resolve_dest(dest), force)
        }
        TlcAction::VerifyProofLibrary { dest } => proof_library::do_verify(&resolve_dest(dest)),
    }
}

/// Resolve the default proof-library directory (`~/tlaplus/tla-library`).
///
/// Exposed so the TLA-library resolution chain and `ty corpus doctor` name the
/// same path this installer writes.
pub(crate) fn default_proof_library() -> PathBuf {
    proof_library::proof_library_path(&resolve_dest(None))
}

/// Resolve the install directory: explicit `--dest`, else `$HOME/tlaplus`.
fn resolve_dest(dest: Option<PathBuf>) -> PathBuf {
    if let Some(d) = dest {
        return d;
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join("tlaplus")
}

fn tlc_jar_path(dest: &Path) -> PathBuf {
    dest.join(TLC_JAR_NAME)
}

fn community_modules_path(dest: &Path) -> PathBuf {
    dest.join(COMMUNITY_MODULES_NAME)
}

/// Probe whether the TLC jar (`tytools.jar`) is already installed at the default
/// dir. Returns the resolved jar path and whether it exists. Used by
/// `ty supremacy reproduce` to decide whether to auto-install.
pub(crate) fn probe_default() -> (PathBuf, bool) {
    let jar = tlc_jar_path(&resolve_dest(None));
    let present = jar.is_file();
    (jar, present)
}

fn do_install(dest: Option<PathBuf>, force: bool, with_proof_library: bool) -> Result<()> {
    let dest = resolve_dest(dest);
    let tlc_jar = tlc_jar_path(&dest);
    let community = community_modules_path(&dest);

    let jars_present = tlc_jar.is_file() && community.is_file();
    if jars_present && !force {
        println!(
            "TLC already installed at {} (tytools.jar + CommunityModules.jar); use --force to re-install",
            dest.display()
        );
    } else {
        std::fs::create_dir_all(&dest)
            .with_context(|| format!("creating TLC install dir {}", dest.display()))?;

        install_tlc_jar(&tlc_jar)?;
        install_community_modules(&community)?;

        println!("TLC installed:");
        println!("  TLC jar:           {}", tlc_jar.display());
        println!("  CommunityModules:  {}", community.display());
    }

    if with_proof_library {
        proof_library::do_install(&dest, force)?;
    } else {
        // Say the quiet part out loud: these two jars alone do not cover the
        // corpus. Silence here is what let a first-party stub mediate 18% of it.
        println!(
            "note: 25 of the 141 eligible corpus rows also need the TLA+ proof library\n      \
             (TLAPS / FiniteSetTheorems / NaturalsInduction). Install it with\n      \
             `ty install-tlc proof-library`, then check with `ty corpus doctor`."
        );
    }

    println!(
        "now run: ty supremacy compare --examples-dir ~/tlaplus-examples/specifications <spec>.tla"
    );
    Ok(())
}

/// Download the nightly `tla2tools.jar` into `<dest>/tytools.jar`, then
/// verify-functional it (`tlc2.TLC -h` prints usage, no NoClassDefFoundError).
fn install_tlc_jar(tlc_jar: &Path) -> Result<()> {
    let dl_dir = std::env::temp_dir().join("ty-tlc-dl");
    let _ = std::fs::remove_dir_all(&dl_dir);
    std::fs::create_dir_all(&dl_dir).with_context(|| format!("creating {}", dl_dir.display()))?;
    let tmp = dl_dir.join("tla2tools.jar");

    eprintln!("downloading TLC (nightly tla2tools.jar) from {TLA2TOOLS_NIGHTLY_URL}");
    run(
        Command::new("curl")
            .args(["-fsSL", "--retry", "3", "-o"])
            .arg(&tmp)
            .arg(TLA2TOOLS_NIGHTLY_URL),
        "curl download of tla2tools.jar",
    )?;

    // Move into place before the functional check so the check exercises the
    // actually-installed file.
    std::fs::rename(&tmp, tlc_jar)
        .or_else(|_| std::fs::copy(&tmp, tlc_jar).map(|_| ()))
        .with_context(|| format!("installing TLC jar to {}", tlc_jar.display()))?;
    let _ = std::fs::remove_dir_all(&dl_dir);

    // Print the sha256 of the nightly jar for the user's record (NOT pinned).
    match sha256_hex(tlc_jar) {
        Ok(hex) => eprintln!("installed TLC jar sha256: {hex} (nightly; not pinned)"),
        Err(e) => eprintln!("note: could not compute TLC jar sha256: {e}"),
    }

    verify_tlc_functional(tlc_jar)?;
    Ok(())
}

/// Run `java -cp <jar> tlc2.TLC -h` and require it prints usage cleanly. If
/// `java` is absent, print a clear JDK hint but do NOT fail the install.
fn verify_tlc_functional(tlc_jar: &Path) -> Result<()> {
    if !java_available() {
        eprintln!(
            "warning: `java` not found on PATH — installed TLC but could not verify it runs. \
             TLC needs a JDK (Java >= 11); install one and re-run `ty install-tlc verify`."
        );
        return Ok(());
    }
    let out = Command::new("java")
        .arg("-cp")
        .arg(tlc_jar)
        .args(["tlc2.TLC", "-h"])
        .output()
        .context("running `java -cp <jar> tlc2.TLC -h` to verify TLC")?;
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if combined.contains("NoClassDefFoundError") || combined.contains("ClassNotFoundException") {
        bail!(
            "TLC jar is broken (class loading error from `tlc2.TLC -h`):\n{}",
            combined.lines().take(8).collect::<Vec<_>>().join("\n")
        );
    }
    // `tlc2.TLC -h` exits 0 and prints a usage banner mentioning "Usage" / "TLC".
    let looks_like_usage = combined.contains("Usage") || combined.to_lowercase().contains("tlc");
    if !looks_like_usage {
        bail!(
            "TLC verify-functional check did not print usage (unexpected output):\n{}",
            combined.lines().take(8).collect::<Vec<_>>().join("\n")
        );
    }
    eprintln!("TLC verify-functional OK (`tlc2.TLC -h` prints usage)");
    Ok(())
}

/// Download + sha256-verify the pinned CommunityModules jar into
/// `<dest>/CommunityModules.jar`.
fn install_community_modules(community: &Path) -> Result<()> {
    let dl_dir = std::env::temp_dir().join("ty-cm-dl");
    let _ = std::fs::remove_dir_all(&dl_dir);
    std::fs::create_dir_all(&dl_dir).with_context(|| format!("creating {}", dl_dir.display()))?;
    let tmp = dl_dir.join("CommunityModules-deps.jar");

    eprintln!("downloading CommunityModules from {COMMUNITY_MODULES_URL}");
    run(
        Command::new("curl")
            .args(["-fsSL", "--retry", "3", "-o"])
            .arg(&tmp)
            .arg(COMMUNITY_MODULES_URL),
        "curl download of CommunityModules jar",
    )?;

    verify_sha256(&tmp, COMMUNITY_MODULES_SHA256).context(
        "CommunityModules jar sha256 mismatch — refusing to install a corrupt/altered asset",
    )?;

    std::fs::rename(&tmp, community)
        .or_else(|_| std::fs::copy(&tmp, community).map(|_| ()))
        .with_context(|| format!("installing CommunityModules jar to {}", community.display()))?;
    let _ = std::fs::remove_dir_all(&dl_dir);
    Ok(())
}

fn do_verify(dest: PathBuf) -> Result<()> {
    let tlc_jar = tlc_jar_path(&dest);
    let community = community_modules_path(&dest);
    if !tlc_jar.is_file() {
        bail!(
            "TLC NOT found: {} does not exist. Run `ty install-tlc install`.",
            tlc_jar.display()
        );
    }
    if !community.is_file() {
        bail!(
            "CommunityModules NOT found: {} does not exist. Run `ty install-tlc install`.",
            community.display()
        );
    }
    verify_tlc_functional(&tlc_jar)?;
    println!(
        "TLC OK at {} (tytools.jar + CommunityModules.jar)",
        dest.display()
    );
    Ok(())
}

fn java_available() -> bool {
    Command::new("java")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Verify a file's sha256 against `expected` (hex). Shells out to `shasum`/
/// `sha256sum` per the repo's no-extra-dep convention.
fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    let out = sha256_hex(path)?;
    if !out.eq_ignore_ascii_case(expected) {
        bail!("sha256 mismatch: got {out}, expected {expected}");
    }
    Ok(())
}

pub(crate) fn sha256_hex(path: &Path) -> Result<String> {
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

pub(crate) fn run(cmd: &mut Command, what: &str) -> Result<()> {
    let status = cmd
        .status()
        .with_context(|| format!("failed to spawn for {what}"))?;
    if !status.success() {
        bail!("{what} failed with status {status}");
    }
    Ok(())
}
