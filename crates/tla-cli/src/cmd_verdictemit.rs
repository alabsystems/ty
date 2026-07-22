// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty verdict-emit` — produce a re-checkable `ty.verdict/v1` envelope for a VIOLATED
//! verdict.
//!
//! Self-contained (like `ty parity`): runs `ty check <spec> --output json` (whose
//! output already embeds the counterexample trace), and on an invariant/property
//! violation packages the trace + the embedded spec into a content-addressed envelope
//! that `ty verdict-check` re-validates independently, eval-only. No verification
//! trust rests on this emitter — the trust is in the in-process re-checker.
//!
//! Exits 0 when an envelope is written, 2 when the run produced no violation to
//! certify (Safe / bounded / error), 1 on a usage/IO error.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use tla_check::verdict::{build_envelope, Completeness, ProducerIdentity, ViolationKind};
use tla_check::{Config, CounterexampleInfo};

#[derive(Deserialize)]
struct CheckJson {
    result: ResultJson,
    #[serde(default)]
    counterexample: Option<CounterexampleInfo>,
}

#[derive(Deserialize)]
struct ResultJson {
    status: String,
    #[serde(default)]
    error_type: Option<String>,
    #[serde(default)]
    violated_property: Option<ViolatedProp>,
}

#[derive(Deserialize)]
struct ViolatedProp {
    name: String,
}

/// Run `ty verdict-emit`.
pub(crate) fn cmd_verdictemit(file: &Path, config_path: Option<&Path>, out: &Path) -> Result<()> {
    if !file.is_file() {
        bail!("spec file not found: {}", file.display());
    }
    let spec_src =
        std::fs::read_to_string(file).with_context(|| format!("read spec {}", file.display()))?;

    let cfg_path = resolve_config(file, config_path)?;
    let config_src = std::fs::read_to_string(&cfg_path)
        .with_context(|| format!("read config {}", cfg_path.display()))?;
    let config = Config::parse(&config_src)
        .map_err(|errs| anyhow::anyhow!("config parse failed with {} error(s)", errs.len()))?;

    // Run the model checker via the real `ty check` JSON path (self-contained).
    let exe = std::env::current_exe().context("could not resolve the `ty` executable path")?;
    let json = run_check_json(&exe, file, &cfg_path)?;
    let parsed: CheckJson =
        serde_json::from_str(&json).context("could not parse `ty check --output json` output")?;

    // Determine the violation kind. Only invariant/property violations produce a
    // re-checkable counterexample envelope in v1.
    let et = parsed.result.error_type.as_deref().unwrap_or("");
    let (kind, violated) =
        match (parsed.result.status.as_str(), et) {
            ("error", "invariant_violation") => (
                ViolationKind::Invariant,
                parsed
                    .result
                    .violated_property
                    .as_ref()
                    .map(|v| v.name.clone()),
            ),
            // `ty check --output json` emits error_type "property_violation" for an
            // action-level temporal violation (a StateLevel []P maps to
            // "invariant_violation", handled by the arm above as ViolationKind::Invariant).
            ("error", "property_violation") => (
                ViolationKind::Property,
                parsed
                    .result
                    .violated_property
                    .as_ref()
                    .map(|v| v.name.clone()),
            ),
            ("error", "deadlock") => (ViolationKind::Deadlock, None),
            _ => {
                eprintln!(
                "NO VIOLATION: `ty check` reported status `{}`{} — there is no counterexample to \
                 certify. `ty verdict-emit` only packages invariant/property/deadlock violations.",
                parsed.result.status,
                if et.is_empty() { String::new() } else { format!(" ({et})") }
            );
                std::process::exit(2);
            }
        };

    let Some(counterexample) = parsed.counterexample else {
        bail!("`ty check` reported a violation but emitted no counterexample trace");
    };
    if counterexample.states.is_empty() {
        bail!(
            "`ty check` reported a violation but its counterexample trace is EMPTY — \
             refusing to package an unreplayable envelope (this indicates a checker bug)"
        );
    }

    let env = build_envelope(
        &spec_src,
        Some(&config_src),
        &config,
        kind,
        violated,
        counterexample,
        // v1: the producing run's completeness is not yet plumbed through the JSON;
        // a violation is a finite witness, so completeness does not gate it.
        Completeness::Exhaustive,
        ProducerIdentity::current(),
    );

    std::fs::write(out, env.to_json())
        .with_context(|| format!("write verdict envelope {}", out.display()))?;
    println!(
        "EMITTED: {} envelope for `{}`\n\
         envelope -> {}\n\
         re-check (independent, eval-only) with: ty verdict-check {}",
        env.verdict,
        env.violated.as_deref().unwrap_or("deadlock"),
        out.display(),
        out.display()
    );
    Ok(())
}

fn resolve_config(file: &Path, config: Option<&Path>) -> Result<std::path::PathBuf> {
    match config {
        Some(p) => Ok(p.to_path_buf()),
        None => {
            let mut cfg = file.to_path_buf();
            cfg.set_extension("cfg");
            if !cfg.exists() {
                bail!(
                    "No config specified and {} does not exist; use --config.",
                    cfg.display()
                );
            }
            Ok(cfg)
        }
    }
}

/// Run `ty check <spec> --config <cfg> --output json` and capture stdout.
fn run_check_json(exe: &Path, spec: &Path, cfg: &Path) -> Result<String> {
    let mut child = Command::new(exe)
        .arg("check")
        .arg(spec)
        .arg("--config")
        .arg(cfg)
        .arg("--output")
        .arg("json")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("spawn `ty check`")?;
    let mut out = String::new();
    if let Some(mut stdout) = child.stdout.take() {
        stdout
            .read_to_string(&mut out)
            .context("read `ty check` stdout")?;
    }
    child.wait().context("wait for `ty check`")?;
    Ok(out)
}
