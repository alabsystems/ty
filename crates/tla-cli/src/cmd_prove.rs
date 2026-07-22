// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty prove` — prove safety for ALL reachable states and immediately re-check the proof.
//!
//! Runs the inductive-safety prover (PDR/IC3 + k-induction over the ay backend) and, on
//! success, emits a `ty.cert/v1` certificate AND re-validates it on the spot — so the
//! verdict you get is not "trust me, it's proved" but "proved, and here is the proof,
//! already re-checked independently." Optionally writes the certificate with `--out`
//! for later `ty recheck`. A proof method, unbounded (stronger than any bounded BFS).
//!
//! Exits 0 on PROVED (+ re-check passed), 1 if the emitted proof fails its own re-check,
//! 2 if the spec is not in the inductive-provable class.

use std::path::Path;

use anyhow::{bail, Context, Result};
use tla_check::Config;

use crate::helpers::read_source;

/// Run `ty prove`.
pub(crate) fn cmd_prove(file: &Path, config_path: Option<&Path>, out: Option<&Path>) -> Result<()> {
    let source = read_source(file)?;
    let config_path = resolve_config(file, config_path)?;
    let config_source = std::fs::read_to_string(&config_path)
        .with_context(|| format!("read config {}", config_path.display()))?;
    let config = match Config::parse(&config_source) {
        Ok(c) => c,
        Err(errors) => {
            for err in &errors {
                eprintln!("{}:{}: {}", config_path.display(), err.line(), err);
            }
            bail!("config parse failed with {} error(s)", errors.len());
        }
    };

    #[cfg(feature = "ay")]
    {
        use tla_check::cert::{certify_spec, verify_safety_certificate, CertVerdict};
        // Fail-closed well-formedness gate (certify/check parity) — `ty prove` mints a re-checkable
        // `ty.cert/v1` just like `ty certify`, so it MUST run the same gate `ty check` enforces
        // before announcing PROVED / writing a certificate for a spec `ty check` refuses as
        // ill-formed (a duplicate definition, an undefined config operator). Shared with every other
        // certificate entry point — this is the 5th, previously missed (see `cert_gate`).
        crate::cert_gate::certify_wellformedness_gate(&source, &config, file)?;
        match certify_spec(&source, &config) {
            Some(cert) => {
                println!(
                    "PROVED: invariant `{}` holds for ALL reachable states (unbounded inductive \
                     proof + deadlock-freedom) — stronger than any bounded search.",
                    cert.invariant_j_tla
                );
                if let Some(out) = out {
                    std::fs::write(out, cert.to_json())
                        .with_context(|| format!("write certificate {}", out.display()))?;
                    println!(
                        "certificate -> {} (re-check anytime with: ty recheck {})",
                        out.display(),
                        out.display()
                    );
                }
                // The proof is only as good as its re-check: re-validate immediately.
                let report = verify_safety_certificate(&cert);
                println!(
                    "\n--- immediate independent re-check ---\n{}",
                    report.detail
                );
                match report.verdict {
                    CertVerdict::Accepted => {
                        // Additive, fail-closed second provenance: independently re-prove the
                        // extractable safety obligation with the tla-zenon tableau prover and
                        // re-check its certificate with tla-cert. This can only ADD a line — it
                        // never changes the verdict, exit code, or emitted certificate.
                        match crate::zenon_leg::corroborate_safety(&source, &cert) {
                            crate::zenon_leg::ZenonLeg::KernelChecked { obligation } => {
                                println!(
                                    "\n--- second, independent kernel proof ---\n\
                                     kernel-checked via tla-zenon (first-order tableau) + tla-cert \
                                     (independent certificate re-check): {obligation}"
                                );
                            }
                            crate::zenon_leg::ZenonLeg::Inconclusive => {
                                // The IC3/PDR path + immediate re-check already established PROVED;
                                // zenon simply had nothing extra to corroborate. Stay silent.
                            }
                        }
                        Ok(())
                    }
                    CertVerdict::Rejected => {
                        eprintln!(
                            "BUG: the emitted proof FAILED its own re-check — do not trust it."
                        );
                        std::process::exit(1);
                    }
                    CertVerdict::Inconclusive => std::process::exit(2),
                }
            }
            None => {
                eprintln!(
                    "NOT PROVED: this spec is not in the inductive-safety provable class. The \
                     normal `ty check` verdict still applies; for a re-checkable VIOLATION use \
                     `ty verdict-emit`, and for cross-engine agreement use `ty parity`."
                );
                std::process::exit(2);
            }
        }
    }

    #[cfg(not(feature = "ay"))]
    {
        let _ = (out, &config, &source);
        bail!("`ty prove` requires the `ay` feature (the inductive prover)");
    }
}

fn resolve_config(file: &Path, config_path: Option<&Path>) -> Result<std::path::PathBuf> {
    match config_path {
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
