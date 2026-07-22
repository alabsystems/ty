// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Plan item **B#3** of `docs/kernel-checked-tla-plan.md`: surface the trust-marker census of
//! the **real** kernel environment TY's `Certified` verdicts run over.
//!
//! Clean ships a green "3-axiom" soundness certificate — but it is computed over
//! `Environment::soundness_certificate_env()`, a curated env that is NOT the
//! `Environment::with_prelude()` that `kernel_accepts` actually type-checks against. The real
//! prelude seeds ~300 domain axioms INCLUDING the four polymorphic trust markers
//! (`sorry`/`sorryAx`/`trustedArith`/`trustedAy`, each `{α : Sort u} → α`). Phase 0's gate
//! guarantees no `Certified` term RESTS on a marker; this module makes the env-level exposure
//! VISIBLE so nobody mistakes the curated 3-axiom number for TY's operating trust base.

use clean_kernel::env::is_trust_marker;

/// The honest census of the env `kernel_accepts` runs over. All counts are over
/// `Environment::with_prelude()` — the REAL check env — not a curated certificate env.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelTrustCensus {
    /// Total declarations in the prelude env.
    pub total_declarations: usize,
    /// Theorems (kernel-checked proof values).
    pub theorems: usize,
    /// Theorems whose transitive closure reaches NO domain axiom.
    pub constructive_theorems: usize,
    /// Theorems resting on at least one domain axiom (surfaced, not hidden).
    pub axiom_dependent_theorems: usize,
    /// Axiom declarations (foundational AND domain).
    pub axioms: usize,
    /// Distinct non-foundational (domain) axioms — includes the trust markers.
    pub total_domain_axioms: usize,
    /// The trust markers present in the env (each proves ANY proposition; Phase 0's
    /// gate rejects any `Certified` term whose closure reaches one).
    pub trust_markers: Vec<String>,
    /// The non-marker domain axioms (sorted). A `Certified` term reaching one of
    /// these is admitted but SURFACED (honestly `AxiomDependent`, plan Phase 0).
    pub domain_axioms: Vec<String>,
}

/// Compute the census over the real `with_prelude()` env. Deterministic; a few
/// hundred ms (walks every declaration's axiom closure).
pub fn kernel_trust_census() -> KernelTrustCensus {
    // The shared cached build of the REAL check env (`soundness_report` is `&self` — read-only).
    let env = crate::cleancic::prelude_env();
    let report = env.soundness_report();
    let (trust_markers, domain_axioms): (Vec<_>, Vec<_>) = report
        .domain_axioms
        .iter()
        .partition(|n| is_trust_marker(n));
    KernelTrustCensus {
        total_declarations: report.total_declarations,
        theorems: report.theorems,
        constructive_theorems: report.constructive_theorems,
        axiom_dependent_theorems: report.axiom_dependent_theorems,
        axioms: report.axioms,
        total_domain_axioms: report.total_domain_axioms,
        trust_markers: trust_markers.into_iter().map(|n| n.to_string()).collect(),
        domain_axioms: domain_axioms.into_iter().map(|n| n.to_string()).collect(),
    }
}

/// Per-run tally of the Phase-0 gate's axiom accounting: how many accepted proof terms
/// were fully constructive (empty non-foundational closure) vs. resting on admitted
/// domain axioms — and WHICH axioms. The `AxiomDependent` bucket is admitted (Phase 0
/// blocks only trust markers) but must be SURFACED, never folded into "Certified" silently.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GateAxiomTally {
    /// Accepted terms whose transitive closure reaches no domain axiom.
    pub constructive: usize,
    /// Accepted terms resting on ≥1 admitted domain axiom (honestly `AxiomDependent`).
    pub axiom_dependent: usize,
    /// The distinct domain axioms the accepted terms reached (sorted).
    pub domain_axioms_seen: std::collections::BTreeSet<String>,
}

std::thread_local! {
    static GATE_TALLY: std::cell::RefCell<Option<GateAxiomTally>> =
        const { std::cell::RefCell::new(None) };
}

/// Start tallying gate axiom accounting on this thread (resets any running tally).
pub fn begin_axiom_tally() {
    GATE_TALLY.with(|t| *t.borrow_mut() = Some(GateAxiomTally::default()));
}

/// Stop tallying and return the counts since [`begin_axiom_tally`].
pub fn take_axiom_tally() -> Option<GateAxiomTally> {
    GATE_TALLY.with(|t| t.borrow_mut().take())
}

/// Whether a tally is active (so the gate can skip the extra closure walk otherwise).
pub(crate) fn axiom_tally_active() -> bool {
    GATE_TALLY.with(|t| t.borrow().is_some())
}

/// Fold `extra` into THIS thread's active tally (no-op when none is running).
/// Order-independent (count sums + a sorted-set union) — the parallel chunk driver
/// ([`crate::cleancic`]'s per-source-state completeness legs) merges its workers'
/// per-worker tallies through this, so a run's printed axiom accounting is IDENTICAL
/// to the sequential loop's (every accepted term still counted exactly once).
pub(crate) fn merge_into_active_axiom_tally(extra: GateAxiomTally) {
    GATE_TALLY.with(|t| {
        if let Some(tally) = t.borrow_mut().as_mut() {
            tally.constructive += extra.constructive;
            tally.axiom_dependent += extra.axiom_dependent;
            tally.domain_axioms_seen.extend(extra.domain_axioms_seen);
        }
    });
}

/// Record one accepted term's non-foundational axiom closure (rendered names).
pub(crate) fn note_gate_axioms(deps: Vec<String>) {
    GATE_TALLY.with(|t| {
        if let Some(tally) = t.borrow_mut().as_mut() {
            if deps.is_empty() {
                tally.constructive += 1;
            } else {
                tally.axiom_dependent += 1;
                tally.domain_axioms_seen.extend(deps);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The census runs over the REAL env and must SURFACE the four trust markers the
    /// prelude seeds (the exposure Phase 0's per-term gate closes). If a Clean change
    /// ever removes the markers from `with_prelude`, this pins the (welcome) diff; if
    /// it ever hides them from the census, that is a dishonesty regression.
    #[test]
    fn census_surfaces_the_preludes_trust_markers() {
        let census = kernel_trust_census();
        for marker in ["sorry", "sorryAx", "trustedArith", "trustedAy"] {
            assert!(
                census.trust_markers.iter().any(|m| m == marker),
                "the real prelude env seeds `{marker}` — the census must SURFACE it \
                 (got: {:?})",
                census.trust_markers
            );
        }
        assert!(
            census.total_domain_axioms > census.trust_markers.len(),
            "the real env carries domain axioms beyond the markers; the curated \
             3-axiom certificate number must not be mistaken for this env"
        );
        assert!(
            census.total_declarations > 1000,
            "with_prelude is a real, big env"
        );
    }
}
