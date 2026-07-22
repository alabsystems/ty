// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! CLI vacuity-gate policy: `--allow-vacuous=<class>[,...]` and
//! `--strict-vacuity`. Applied once, after the run and before result reporting,
//! at the single chokepoint in `cmd_check`.
//!
//! Design: TRUST_VACUITY_GATE §1.A. The core checker emits the *raw* signal
//! (a `Vacuous` verdict for V1; `stats.vacuity_warnings` for V2/V3); the policy
//! (downgrade a named class to a recorded WARNING, or promote WARNINGs to the
//! hard verdict) is the caller's choice and lives here.

use std::collections::HashSet;

use tla_check::{CheckResult, VacuityClass, VacuityReason};

/// Parsed `--allow-vacuous` / `--strict-vacuity` policy.
#[derive(Debug, Clone, Default)]
pub(crate) struct VacuityPolicy {
    /// Classes downgraded from a hard VACUOUS verdict to a recorded WARNING.
    allow: HashSet<VacuityClass>,
    /// Promote default-on V2/V3 WARNINGs to the hard VACUOUS verdict (exit 3).
    strict: bool,
}

/// Outcome of applying the policy: the (possibly rewritten) result plus the
/// human-readable lines the CLI should print (warnings + audited downgrades).
pub(crate) struct VacuityPolicyOutcome {
    pub(crate) result: CheckResult,
    pub(crate) lines: Vec<String>,
}

impl VacuityPolicy {
    /// Parse the CLI flags. Unknown class tokens are reported as errors so a
    /// typo'd `--allow-vacuous` never silently fails open.
    pub(crate) fn parse(allow_vacuous: &[String], strict_vacuity: bool) -> anyhow::Result<Self> {
        let mut allow = HashSet::new();
        for token in allow_vacuous {
            // value_delimiter splits on commas, but be defensive about extra
            // whitespace / empty tokens.
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            match VacuityClass::parse(token) {
                Some(c) => {
                    allow.insert(c);
                }
                None => {
                    let known: Vec<&str> = VacuityClass::all().iter().map(|c| c.as_str()).collect();
                    anyhow::bail!(
                        "unknown --allow-vacuous class `{token}` (known: {})",
                        known.join(", ")
                    );
                }
            }
        }
        Ok(VacuityPolicy {
            allow,
            strict: strict_vacuity,
        })
    }

    /// Apply the policy to a finished `CheckResult`.
    ///
    /// - A hard `Vacuous` verdict whose class is in `--allow-vacuous` is
    ///   downgraded to `Success`, with an audited line recording the relaxation.
    /// - When `--strict-vacuity` is set, any default-on V2/V3 WARNING (that is
    ///   NOT downgraded by `--allow-vacuous`) promotes a `Success`/`Vacuous`
    ///   result to a hard `Vacuous` verdict.
    /// - Otherwise V2/V3 WARNINGs are printed but non-fatal.
    pub(crate) fn apply(&self, result: CheckResult) -> VacuityPolicyOutcome {
        let mut lines = Vec::new();

        // Always surface the default-on V2/V3 WARNINGs (subject to downgrade).
        let warnings = result.stats().vacuity_warnings.clone();
        let mut active_warnings = Vec::new();
        for w in &warnings {
            if self.allow.contains(&w.class()) {
                lines.push(format!(
                    "note: vacuity warning downgraded by --allow-vacuous={}: {}",
                    w.class().as_str(),
                    w.message()
                ));
            } else {
                lines.push(format!("Warning: vacuous: {}", w.message()));
                active_warnings.push(w.clone());
            }
        }

        // (1) Downgrade a hard VACUOUS verdict whose class is allowed.
        if let CheckResult::Vacuous { reason, stats } = &result {
            if self.allow.contains(&reason.class()) {
                lines.push(format!(
                    "note: VACUOUS verdict downgraded to warning by --allow-vacuous={}: {}",
                    reason.class().as_str(),
                    reason.message()
                ));
                return VacuityPolicyOutcome {
                    result: CheckResult::Success(stats.clone()),
                    lines,
                };
            }
            // Hard verdict stands.
            return VacuityPolicyOutcome { result, lines };
        }

        // (2) Promote V2/V3 WARNINGs to a hard verdict under --strict-vacuity.
        if self.strict && !active_warnings.is_empty() {
            // Only promote a result that is currently "passing" (Success). A
            // genuine property failure / error keeps its own (more severe)
            // verdict and exit code.
            if let CheckResult::Success(stats) = result {
                let reason = strict_reason_from_warnings(&active_warnings);
                lines.push(
                    "error: --strict-vacuity: promoting vacuity warning(s) to VACUOUS verdict"
                        .to_string(),
                );
                return VacuityPolicyOutcome {
                    result: CheckResult::Vacuous { reason, stats },
                    lines,
                };
            }
        }

        VacuityPolicyOutcome { result, lines }
    }
}

/// Build a single `VacuityReason` from the promoted WARNINGs. Prefers a
/// dead-action reason (V2) when present, else the vacuous-invariant names (V3).
fn strict_reason_from_warnings(warnings: &[tla_check::VacuityWarning]) -> VacuityReason {
    use tla_check::VacuityWarning;
    let mut dead = Vec::new();
    let mut invs = Vec::new();
    for w in warnings {
        match w {
            VacuityWarning::DeadActions(names) => dead.extend(names.iter().cloned()),
            VacuityWarning::AntecedentNeverHolds { invariant }
            | VacuityWarning::ConstantTrueInvariant { invariant } => invs.push(invariant.clone()),
        }
    }
    if !dead.is_empty() {
        VacuityReason::DeadActions(dead)
    } else {
        VacuityReason::VacuousInvariants(invs)
    }
}
