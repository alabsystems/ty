// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty verdict-check` — independently re-check a `ty.verdict/v1` envelope.
//!
//! The envelope is self-contained (it embeds the spec source and the counterexample
//! trace), so this command needs ONLY the envelope file. It re-parses the embedded
//! spec and replays the trace through the tree-walking evaluator: `Init(s0)`, each
//! `Next(s_i, s_{i+1})`, and the named invariant FALSE at the final state. The trusted
//! base is the parser + the evaluator + this checker — NOT the BFS engine, the native
//! backend, or the SMT solver that produced the verdict.
//!
//! Exits 0 on VERIFIED, 1 on REJECTED, 2 on INCONCLUSIVE.

use std::path::Path;

use anyhow::{Context, Result};
use tla_check::verdict::{verify_violation_envelope, VerdictEnvelope, VerdictVerdict};

/// Run `ty verdict-check`.
pub(crate) fn cmd_verdictcheck(envelope: &Path) -> Result<()> {
    let json = std::fs::read_to_string(envelope)
        .with_context(|| format!("read verdict envelope {}", envelope.display()))?;
    let env = VerdictEnvelope::from_json(&json).map_err(anyhow::Error::msg)?;

    let report = verify_violation_envelope(&env);
    println!("{}", report.detail);

    match report.verdict {
        VerdictVerdict::Verified => Ok(()),
        VerdictVerdict::Rejected => std::process::exit(1),
        VerdictVerdict::Inconclusive => std::process::exit(2),
    }
}
