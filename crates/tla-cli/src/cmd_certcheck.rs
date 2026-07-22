// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty cert-check` subcommand: independently re-check a `ty.cert/v1` certificate.
//!
//! The certificate is self-contained (it carries its own spec source and the
//! per-obligation proof bundles), so this command needs ONLY the certificate
//! file. Acceptance rests on the EXTERNAL proof re-check (Leg D): each SMT
//! obligation's embedded proof is re-validated by AY's audited `check_proof_strict`
//! with NO solver search, bound to the obligation by an assume-coverage gate. The
//! explicit-state eval oracle re-derives reachable states from the certificate's
//! spec as an engine-diverse cross-check, so a tampered invariant is rejected even
//! if its digest was recomputed to match. Exits 0 on VERIFIED, 1 on REJECTED, 2 on
//! INCONCLUSIVE.

use std::path::Path;

use anyhow::{Context, Result};
use tla_check::cert::{verify_safety_certificate, CertVerdict, SafetyCertificate};

/// Run `ty cert-check`. With `carcara`, additionally re-checks each embedded proof
/// with carcara (a separate, independently-implemented Alethe checker) for
/// N-version redundancy; a carcara `holey`/`invalid` is an additional REJECT.
pub(crate) fn cmd_certcheck(cert_file: &Path, carcara: bool) -> Result<()> {
    let json = std::fs::read_to_string(cert_file)
        .with_context(|| format!("read certificate {}", cert_file.display()))?;
    let cert = SafetyCertificate::from_json(&json).map_err(anyhow::Error::msg)?;

    let report = verify_safety_certificate(&cert);
    println!("{}", report.detail);

    // Third-party N-version re-check (carcara). `Some(false)` (a holey/invalid
    // proof) is an additional REJECT; `None` (carcara absent) never flips an
    // accept. Runs regardless of the `ay` feature (pure Alethe text re-check).
    let carcara_ok = if carcara {
        crate::cmd_certexport::carcara_recheck_certificate(&json)?
    } else {
        Some(true)
    };
    if carcara_ok == Some(false) {
        eprintln!("REJECTED: carcara (independent Alethe checker) rejected an embedded proof");
        std::process::exit(1);
    }

    // Exit codes: 0 = VERIFIED, 1 = REJECTED (certificate is definitively bad),
    // 2 = INCONCLUSIVE (could not re-validate — e.g. this binary lacks the `ay`
    // feature so the AY inductive proof cannot be re-discharged). INCONCLUSIVE is
    // never a false accept and is distinct from the certificate being invalid.
    match report.verdict {
        CertVerdict::Accepted => Ok(()),
        CertVerdict::Rejected => std::process::exit(1),
        CertVerdict::Inconclusive => {
            #[cfg(not(feature = "ay"))]
            eprintln!(
                "INCONCLUSIVE: this `ty` was built WITHOUT the `ay` feature, so the AY \
                 inductive proof cannot be re-discharged. Rebuild with `--features ay` to \
                 re-validate. (This is NOT a verdict on the certificate.)"
            );
            std::process::exit(2);
        }
    }
}
