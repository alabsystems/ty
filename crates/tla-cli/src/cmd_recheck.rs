// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty recheck` — the unified, minimal re-checker.
//!
//! One command that re-validates any TY assurance artifact by its `schema` tag,
//! dispatching to the appropriate independent verifier (`ty.verdict/v1` →
//! eval-only trace replay; `ty.cert/v1` → inductive-proof re-check). `ty recheck --tcb`
//! prints the declared TRUSTED COMPUTING BASE — the small named kernel the north star
//! demands: a large untrusted prover, a small trusted re-checker the user holds.
//!
//! Exits 0 on VERIFIED, 1 on REJECTED, 2 on INCONCLUSIVE.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use tla_check::cert::{verify_safety_certificate, CertVerdict, SafetyCertificate};
use tla_check::verdict::{verify_violation_envelope, VerdictEnvelope, VerdictVerdict};

#[derive(Deserialize)]
struct SchemaPeek {
    #[serde(default)]
    schema: String,
}

/// Run `ty recheck`.
pub(crate) fn cmd_recheck(artifact: Option<&Path>, show_tcb: bool) -> Result<()> {
    if show_tcb {
        print_tcb();
        if artifact.is_none() {
            return Ok(());
        }
    }
    let artifact =
        artifact.context("provide an artifact JSON, or use --tcb to print the trusted base")?;
    let json = std::fs::read_to_string(artifact)
        .with_context(|| format!("read artifact {}", artifact.display()))?;
    let peek: SchemaPeek =
        serde_json::from_str(&json).context("artifact is not valid JSON with a `schema` field")?;

    match peek.schema.as_str() {
        "ty.verdict/v1" => {
            let env = VerdictEnvelope::from_json(&json).map_err(anyhow::Error::msg)?;
            let report = verify_violation_envelope(&env);
            println!("{}", report.detail);
            match report.verdict {
                VerdictVerdict::Verified => Ok(()),
                VerdictVerdict::Rejected => std::process::exit(1),
                VerdictVerdict::Inconclusive => std::process::exit(2),
            }
        }
        "ty.cert/v1" => {
            let cert = SafetyCertificate::from_json(&json).map_err(anyhow::Error::msg)?;
            let report = verify_safety_certificate(&cert);
            println!("{}", report.detail);
            match report.verdict {
                CertVerdict::Accepted => Ok(()),
                CertVerdict::Rejected => std::process::exit(1),
                CertVerdict::Inconclusive => std::process::exit(2),
            }
        }
        "" => bail!("artifact has no `schema` field — not a TY assurance artifact"),
        other => bail!(
            "unknown artifact schema `{other}` — `ty recheck` understands `ty.verdict/v1` \
             and `ty.cert/v1`"
        ),
    }
}

fn print_tcb() {
    println!(
        "ty recheck — declared TRUSTED COMPUTING BASE\n\
         \n\
         The re-check trusts ONLY a small, named kernel — never the model checker, the\n\
         native JIT, or the SMT solver's search that PRODUCED the verdict.\n\
         \n\
         ty.verdict/v1 (replayable counterexample):\n\
         \x20 - tla-core      parse + lower the embedded spec\n\
         \x20 - tla-eval      evaluate Init / Next / the invariant against trace states\n\
         \x20 - ty.verdict    this re-checker (replay + terminal-violation leg)\n\
         \x20 NOT trusted: the BFS engine, trust-cg native backend, or ay/SMT search.\n\
         \n\
         ty.cert/v1 (inductive-safety proof):\n\
         \x20 - tla-core      parse + lower the embedded spec\n\
         \x20 - tla-eval      explicit-state eval oracle (the soundness spine)\n\
         \x20 - ay proof checker  re-discharge each obligation with NO solver search\n\
         \x20 - (optional) carcara / Lean: independent N-version proof re-check\n\
         \x20 NOT trusted: the SMT solver's search that synthesized the invariant.\n"
    );
}
