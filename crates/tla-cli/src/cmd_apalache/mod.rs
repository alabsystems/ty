// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty install-apalache` subcommand (alias: `ty apalache`): download and install Apalache (the symbolic TLA+
//! model checker) so the Apalache-vs-TY comparison runs with no manual setup.
//!
//! Installs the pinned release into `~/apalache/` (override with `--dest`) so
//! that `<dest>/bin/apalache-mc` resolves. The release tarball is immutable, so
//! its sha256 is pinned and verified before extraction. After install we
//! verify-functional it by running `apalache-mc version` (needs a JDK; we print
//! a clear hint if `java`/`JAVA_HOME` is absent but still complete the install).
//!
//! Follows the repo convention of shelling out to `curl`/`tar` rather than
//! adding an HTTP client dependency (cf. `cmd_corpus`).

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};

use crate::cli_schema::ApalacheAction;

/// Pinned Apalache release (immutable asset).
const APALACHE_VERSION: &str = "0.58.0";
const APALACHE_URL: &str =
    "https://github.com/apalache-mc/apalache/releases/download/v0.58.0/apalache-0.58.0.tgz";
/// sha256 of `apalache-0.58.0.tgz` (verified before extraction).
const APALACHE_SHA256: &str = "55b4da129140b3b6b4106b31eddf36b5f49d896bf2ec8b4cf81c93bc37b9b3d7";
/// Top-level directory inside the tarball.
const APALACHE_TARBALL_ROOT: &str = "apalache-0.58.0";

/// Dispatch the `ty install-apalache <action>` subcommand.
pub(crate) fn cmd_apalache(action: ApalacheAction) -> Result<()> {
    match action {
        ApalacheAction::Install { dest, force } => do_install(dest, force),
        ApalacheAction::Path { dest } => {
            println!("{}", resolve_dest(dest).display());
            Ok(())
        }
        ApalacheAction::Verify { dest } => do_verify(resolve_dest(dest)),
    }
}

/// Resolve the install directory: explicit `--dest`, else `$HOME/apalache`.
fn resolve_dest(dest: Option<PathBuf>) -> PathBuf {
    if let Some(d) = dest {
        return d;
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join("apalache")
}

fn launcher_path(dest: &Path) -> PathBuf {
    dest.join("bin").join("apalache-mc")
}

/// Probe whether the Apalache launcher is already installed at the default dir.
/// Returns the resolved launcher path and whether it exists. Used by
/// `ty supremacy reproduce` to decide whether to auto-install.
pub(crate) fn probe_default() -> (PathBuf, bool) {
    let launcher = launcher_path(&resolve_dest(None));
    let present = launcher.is_file();
    (launcher, present)
}

fn do_install(dest: Option<PathBuf>, force: bool) -> Result<()> {
    let dest = resolve_dest(dest);
    let launcher = launcher_path(&dest);

    if launcher.is_file() && !force {
        println!(
            "Apalache already installed at {} ({}); use --force to re-install",
            dest.display(),
            launcher.display()
        );
        return Ok(());
    }

    let dl_dir = std::env::temp_dir().join("ty-apalache-dl");
    let _ = std::fs::remove_dir_all(&dl_dir);
    std::fs::create_dir_all(&dl_dir).with_context(|| format!("creating {}", dl_dir.display()))?;
    let tarball = dl_dir.join(format!("apalache-{APALACHE_VERSION}.tgz"));

    eprintln!("downloading Apalache {APALACHE_VERSION} from {APALACHE_URL}");
    run(
        Command::new("curl")
            .args(["-fsSL", "--retry", "3", "-o"])
            .arg(&tarball)
            .arg(APALACHE_URL),
        "curl download of Apalache tarball",
    )?;

    verify_sha256(&tarball, APALACHE_SHA256).context(
        "Apalache tarball sha256 mismatch — refusing to extract a corrupt/altered asset",
    )?;

    // Extract into the temp dir; the tarball unpacks to `apalache-0.58.0/`.
    eprintln!("extracting Apalache");
    run(
        Command::new("tar")
            .arg("-xzf")
            .arg(&tarball)
            .arg("-C")
            .arg(&dl_dir),
        "tar extract of Apalache tarball",
    )?;
    let extracted = dl_dir.join(APALACHE_TARBALL_ROOT);
    if !extracted.is_dir() {
        bail!(
            "Apalache tarball did not unpack to expected {} ",
            extracted.display()
        );
    }

    // Move the extracted tree into place. Clear any stale dest first (on
    // --force) so the rename/copy lands cleanly.
    let _ = std::fs::remove_dir_all(&dest);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating parent of install dir {}", parent.display()))?;
    }
    if std::fs::rename(&extracted, &dest).is_err() {
        // Cross-filesystem rename can fail; fall back to a recursive copy.
        run(
            Command::new("cp").arg("-R").arg(&extracted).arg(&dest),
            "copy extracted Apalache into install dir",
        )?;
    }
    let _ = std::fs::remove_dir_all(&dl_dir);

    let launcher = launcher_path(&dest);
    if !launcher.is_file() {
        bail!(
            "install completed but launcher missing at {}",
            launcher.display()
        );
    }
    ensure_executable(&launcher)?;

    println!("Apalache {APALACHE_VERSION} installed:");
    println!("  launcher: {}", launcher.display());

    verify_apalache_functional(&launcher)?;
    Ok(())
}

fn do_verify(dest: PathBuf) -> Result<()> {
    let launcher = launcher_path(&dest);
    if !launcher.is_file() {
        bail!(
            "Apalache NOT found: {} does not exist. Run `ty install-apalache install`.",
            launcher.display()
        );
    }
    ensure_executable(&launcher)?;
    verify_apalache_functional(&launcher)?;
    println!("Apalache OK at {} ({})", dest.display(), launcher.display());
    Ok(())
}

/// Run `<launcher> version` and require it succeeds. If a JDK is absent, print a
/// clear hint but do NOT fail the install (the binary is still in place).
fn verify_apalache_functional(launcher: &Path) -> Result<()> {
    if !java_available() {
        eprintln!(
            "warning: no JDK detected (`java` not on PATH and JAVA_HOME unset) — installed \
             Apalache but could not verify it runs. Apalache needs a JDK (Java >= 17); install \
             one and re-run `ty install-apalache verify`."
        );
        return Ok(());
    }
    let out = Command::new(launcher)
        .arg("version")
        .output()
        .with_context(|| {
            format!(
                "running `{} version` to verify Apalache",
                launcher.display()
            )
        })?;
    if !out.status.success() {
        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        bail!(
            "Apalache verify-functional check failed (`apalache-mc version` exited {}):\n{}",
            out.status,
            combined.lines().take(8).collect::<Vec<_>>().join("\n")
        );
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    eprintln!(
        "Apalache verify-functional OK (`apalache-mc version`): {}",
        stdout.lines().next().unwrap_or("").trim()
    );
    Ok(())
}

fn java_available() -> bool {
    if std::env::var_os("JAVA_HOME").is_some() {
        return true;
    }
    Command::new("java")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Ensure the launcher has the executable bit set (tar usually preserves it,
/// but be defensive).
fn ensure_executable(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let meta = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
        let mut perms = meta.permissions();
        let mode = perms.mode();
        if mode & 0o111 == 0 {
            perms.set_mode(mode | 0o755);
            std::fs::set_permissions(path, perms)
                .with_context(|| format!("chmod +x {}", path.display()))?;
        }
    }
    let _ = path;
    Ok(())
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

fn sha256_hex(path: &Path) -> Result<String> {
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
