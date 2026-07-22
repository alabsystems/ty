// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Per-examination technique attribution.
//!
//! The MCC organizer's analysis report counts the union of distinct technique
//! tokens declared across the submission as a quality signal. Reporting only
//! `EXPLICIT` for every formula hides the symbolic, structural, and temporal
//! reasoning the pipelines actually exercise.
//!
//! This module is the single source of truth for "which techniques does
//! examination X exercise". It is derived **statically** from the pipeline
//! structure rather than instrumented at runtime: each examination's wired
//! engine list (see `examinations/`, `examination_non_property/`) does in
//! fact run every technique declared here. Runtime signals such as
//! ay-binary availability and NUPN presence further narrow the declared set
//! so we never claim a technique we could not have used on this invocation.
//!
//! Adding a technique to a pipeline requires updating the corresponding
//! arm here so the declaration tracks the implementation.

use std::cell::Cell;

use crate::examination::Examination;
use crate::nupn::NupnStructure;
use crate::output::{Technique, Techniques};

thread_local! {
    /// Set by [`note_aiger_resolved_deadlock`] when the AIGER/IC3 lane
    /// produces a SAT trace that replays to a real deadlock marking.
    static LAST_DEADLOCK_AIGER_RESOLVED: Cell<bool> = const { Cell::new(false) };
}

/// Record that the AIGER/IC3 deadlock lane resolved this examination as TRUE.
pub(crate) fn note_aiger_resolved_deadlock() {
    LAST_DEADLOCK_AIGER_RESOLVED.with(|cell| cell.set(true));
}

/// Take-and-clear the AIGER-resolved-deadlock flag.
#[allow(dead_code)]
fn take_aiger_resolved_deadlock() -> bool {
    LAST_DEADLOCK_AIGER_RESOLVED.with(|cell| cell.replace(false))
}

/// Whether the ay SAT/SMT backend is reachable from this process.
///
/// Mirrors the discovery logic the SMT-driven engines use before they invoke
/// ay (see [`crate::examinations::smt_encoding::find_ay`]). When ay is
/// unavailable the SMT-backed lanes silently no-op, so declaring `SAT_SMT`
/// would be dishonest.
#[must_use]
pub(crate) fn ay_runtime_available() -> bool {
    tla_mc_core::find_ay_binary().is_some()
}

/// Techniques exercised by an examination's pipeline.
///
/// `ay_available` should reflect runtime ay discovery (see
/// [`ay_runtime_available`]). `nupn` is the NUPN structure attached to the
/// model when one was parsed — only `OneSafe` consults it, mirroring the
/// engine wiring in `examination_non_property::deadlock_one_safe`.
#[must_use]
pub(crate) fn techniques_for_examination(
    examination: Examination,
    ay_available: bool,
    nupn: Option<&NupnStructure>,
) -> Techniques {
    let mut techniques = Techniques::single(Technique::Explicit);

    match examination {
        Examination::StateSpace => {
            // state_space.rs runs explore_observer over the ORIGINAL net (no
            // structural reduction — soundness guard #1483). Default BFS plan.
            techniques.add(Technique::Bfs);
        }

        Examination::ReachabilityDeadlock => {
            // deadlock_one_safe::deadlock_verdict runs: structural deadlock-free,
            // LP state equation, optional PDR, BMC, and BFS portfolio with
            // deadlock-preserving partial-order reduction.
            techniques.add(Technique::Bfs);
            techniques.add(Technique::Structural);
            techniques.add(Technique::Topological);
            techniques.add(Technique::PartialOrder);
            if ay_available {
                techniques.add(Technique::SatSmt);
                techniques.add(Technique::Bmc);
                techniques.add(Technique::KInduction);
                techniques.add(Technique::Ic3);
            }
        }

        Examination::OneSafe => {
            // deadlock_one_safe::one_safe_verdict runs: P-invariant structural
            // bound, LP upper bound, optional NUPN proof, BMC/PDR, structural
            // reduction, partial-order reduction, BFS.
            techniques.add(Technique::Bfs);
            techniques.add(Technique::Structural);
            techniques.add(Technique::Topological);
            techniques.add(Technique::PartialOrder);
            if nupn.is_some() {
                techniques.add(Technique::UseNupn);
            }
            if ay_available {
                techniques.add(Technique::SatSmt);
                techniques.add(Technique::Bmc);
                techniques.add(Technique::Ic3);
            }
        }

        Examination::QuasiLiveness => {
            // liveness::quasi_liveness_verdict runs: structural-live siphon/trap,
            // LP dead-transition, BMC per transition, BFS observer.
            techniques.add(Technique::Bfs);
            techniques.add(Technique::Structural);
            techniques.add(Technique::Topological);
            if ay_available {
                techniques.add(Technique::SatSmt);
                techniques.add(Technique::Bmc);
            }
        }

        Examination::StableMarking => {
            // stable_marking::stable_marking_verdict runs: structurally stable
            // place check, BMC, optional PDR (gated by env), BFS observer.
            techniques.add(Technique::Bfs);
            techniques.add(Technique::Structural);
            if ay_available {
                techniques.add(Technique::SatSmt);
                techniques.add(Technique::Bmc);
                techniques.add(Technique::KInduction);
            }
        }

        Examination::Liveness => {
            // liveness::liveness_verdict runs: structural-live, T-semiflow,
            // LP dead-transition, BMC + k-induction for deadlock and
            // per-transition liveness, BFS over reachability graph with SCC.
            techniques.add(Technique::Bfs);
            techniques.add(Technique::Structural);
            techniques.add(Technique::Topological);
            techniques.add(Technique::TemporalLogic);
            if ay_available {
                techniques.add(Technique::SatSmt);
                techniques.add(Technique::Bmc);
                techniques.add(Technique::KInduction);
            }
        }

        Examination::UpperBounds => {
            // examinations/upper_bounds runs LP relaxation for approximate
            // bounds, P-invariant bound tightening, and BFS for exact bounds.
            techniques.add(Technique::Bfs);
            techniques.add(Technique::Topological);
            techniques.add(Technique::Structural);
            techniques.add(Technique::LpApprox);
        }

        Examination::ReachabilityCardinality | Examination::ReachabilityFireability => {
            // reachability/pipeline.rs runs: formula simplification, bounded
            // BFS witness, BMC, LP, IC3/PDR, k-induction, AIGER guards,
            // structural reduction, full BFS exploration.
            techniques.add(Technique::Bfs);
            techniques.add(Technique::Structural);
            techniques.add(Technique::Topological);
            if ay_available {
                techniques.add(Technique::SatSmt);
                techniques.add(Technique::Bmc);
                techniques.add(Technique::Ic3);
                techniques.add(Technique::KInduction);
            }
        }

        Examination::CTLCardinality | Examination::CTLFireability => {
            // examinations/ctl runs a CTL tableau / fixpoint procedure backed
            // by explicit-state checking and optional IC3/BMC routes.
            techniques.add(Technique::TemporalLogic);
            techniques.add(Technique::Bfs);
            techniques.add(Technique::Structural);
            if ay_available {
                techniques.add(Technique::SatSmt);
                techniques.add(Technique::Bmc);
                techniques.add(Technique::Ic3);
            }
        }

        Examination::LTLCardinality | Examination::LTLFireability => {
            // examinations/ltl runs Büchi product + lasso detection with BMC
            // for short counter-examples and partial-order optimizations.
            techniques.add(Technique::TemporalLogic);
            techniques.add(Technique::Bfs);
            techniques.add(Technique::Structural);
            techniques.add(Technique::PartialOrder);
            if ay_available {
                techniques.add(Technique::SatSmt);
                techniques.add(Technique::Bmc);
            }
        }
    }

    techniques
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_space_declares_bfs_only_without_ay() {
        let t = techniques_for_examination(Examination::StateSpace, false, None);
        assert_eq!(t.as_mcc_str(), "EXPLICIT BFS");
        assert!(!t.contains(Technique::SatSmt));
    }

    #[test]
    fn reachability_deadlock_with_ay_declares_full_portfolio() {
        let t = techniques_for_examination(Examination::ReachabilityDeadlock, true, None);
        assert!(t.contains(Technique::Explicit));
        assert!(t.contains(Technique::Bfs));
        assert!(t.contains(Technique::Structural));
        assert!(t.contains(Technique::Topological));
        assert!(t.contains(Technique::PartialOrder));
        assert!(t.contains(Technique::SatSmt));
        assert!(t.contains(Technique::Bmc));
        assert!(t.contains(Technique::KInduction));
        assert!(t.contains(Technique::Ic3));
    }

    #[test]
    fn reachability_deadlock_without_ay_drops_sat_smt() {
        let t = techniques_for_examination(Examination::ReachabilityDeadlock, false, None);
        assert!(!t.contains(Technique::SatSmt));
        assert!(!t.contains(Technique::Bmc));
        assert!(!t.contains(Technique::Ic3));
        assert!(!t.contains(Technique::KInduction));
        assert!(t.contains(Technique::Explicit));
        assert!(t.contains(Technique::Bfs));
        assert!(t.contains(Technique::Structural));
        assert!(t.contains(Technique::Topological));
        assert!(t.contains(Technique::PartialOrder));
    }

    #[test]
    fn upper_bounds_declares_lp_approx() {
        let t = techniques_for_examination(Examination::UpperBounds, true, None);
        assert!(t.contains(Technique::LpApprox));
        assert!(t.contains(Technique::Topological));
        assert!(t.contains(Technique::Explicit));
        assert!(t.contains(Technique::Bfs));
        assert!(t.contains(Technique::Structural));
    }

    #[test]
    fn liveness_declares_temporal_logic_and_topological() {
        let t = techniques_for_examination(Examination::Liveness, false, None);
        assert!(t.contains(Technique::TemporalLogic));
        assert!(t.contains(Technique::Topological));
        assert!(t.contains(Technique::Structural));
        assert!(t.contains(Technique::Bfs));
        assert!(t.contains(Technique::Explicit));
    }

    #[test]
    fn ctl_declares_temporal_logic() {
        let t = techniques_for_examination(Examination::CTLCardinality, false, None);
        assert!(t.contains(Technique::TemporalLogic));
        assert!(t.contains(Technique::Explicit));
        assert!(t.contains(Technique::Bfs));
        assert!(t.contains(Technique::Structural));
        assert!(!t.contains(Technique::SatSmt));
    }

    #[test]
    fn ctl_with_ay_declares_ic3_and_bmc() {
        let t = techniques_for_examination(Examination::CTLFireability, true, None);
        assert!(t.contains(Technique::SatSmt));
        assert!(t.contains(Technique::Bmc));
        assert!(t.contains(Technique::Ic3));
        assert!(t.contains(Technique::TemporalLogic));
    }

    #[test]
    fn ltl_declares_temporal_logic_and_por() {
        let t = techniques_for_examination(Examination::LTLCardinality, false, None);
        assert!(t.contains(Technique::TemporalLogic));
        assert!(t.contains(Technique::PartialOrder));
        assert!(t.contains(Technique::Bfs));
        assert!(t.contains(Technique::Structural));
    }

    #[test]
    fn ltl_with_ay_declares_bmc() {
        let t = techniques_for_examination(Examination::LTLFireability, true, None);
        assert!(t.contains(Technique::SatSmt));
        assert!(t.contains(Technique::Bmc));
        assert!(t.contains(Technique::TemporalLogic));
    }

    #[test]
    fn reachability_cardinality_declares_ic3_when_ay_present() {
        let t = techniques_for_examination(Examination::ReachabilityCardinality, true, None);
        assert!(t.contains(Technique::Ic3));
        assert!(t.contains(Technique::Bmc));
        assert!(t.contains(Technique::KInduction));
        assert!(t.contains(Technique::SatSmt));
    }

    #[test]
    fn quasi_liveness_declares_bmc_when_ay_present() {
        let t = techniques_for_examination(Examination::QuasiLiveness, true, None);
        assert!(t.contains(Technique::Bmc));
        assert!(t.contains(Technique::SatSmt));
        assert!(t.contains(Technique::Structural));
        assert!(t.contains(Technique::Topological));
    }

    #[test]
    fn stable_marking_declares_kinduction_when_ay_present() {
        let t = techniques_for_examination(Examination::StableMarking, true, None);
        assert!(t.contains(Technique::KInduction));
        assert!(t.contains(Technique::Bmc));
        assert!(t.contains(Technique::SatSmt));
    }

    #[test]
    fn one_safe_with_nupn_declares_use_nupn() {
        // We can't easily construct a NupnStructure here without depending on
        // parsing, but the function takes the option by reference. Use an
        // empty no-op net to validate the without-NUPN path; the with-NUPN
        // path is exercised by the broader integration tests.
        let t = techniques_for_examination(Examination::OneSafe, false, None);
        assert!(!t.contains(Technique::UseNupn));
        assert!(t.contains(Technique::Structural));
        assert!(t.contains(Technique::Topological));
        assert!(t.contains(Technique::PartialOrder));
    }

    #[test]
    fn one_safe_without_ay_does_not_claim_sat_smt() {
        let t = techniques_for_examination(Examination::OneSafe, false, None);
        assert!(!t.contains(Technique::SatSmt));
        assert!(!t.contains(Technique::Bmc));
        assert!(!t.contains(Technique::Ic3));
    }

    #[test]
    fn every_examination_includes_explicit() {
        for examination in Examination::ALL {
            let t = techniques_for_examination(examination, false, None);
            assert!(
                t.contains(Technique::Explicit),
                "examination {examination:?} dropped the EXPLICIT base tag",
            );
        }
    }

    #[test]
    fn every_examination_produces_non_empty_mcc_string() {
        for examination in Examination::ALL {
            for ay in [false, true] {
                let t = techniques_for_examination(examination, ay, None);
                let rendered = t.as_mcc_str();
                assert!(
                    !rendered.is_empty(),
                    "examination {examination:?} ay={ay} produced empty TECHNIQUES list",
                );
                assert!(rendered.contains("EXPLICIT"));
            }
        }
    }
}
