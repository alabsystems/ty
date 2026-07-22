// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared tracker and preparation types for reachability examinations.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::examination::ExaminationValue;
use crate::examinations::certified::{
    BoolVerdict, CertifiedExaminationRecord, Evidence, EvidenceSet, UnknownReason,
};
use crate::examinations::reachability_witness::WitnessValidationTarget;
use crate::model::PropertyAliases;
use crate::output::{Technique, Verdict};
use crate::property_xml::{Formula, PathQuantifier, Property, ReachabilityFormula};
use crate::resolved_predicate::{
    count_unresolved_with_aliases, resolve_predicate_with_aliases, ResolvedPredicate,
};

/// Which pipeline phase resolved a reachability property.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum ReachabilityResolutionSource {
    Bmc,
    Lp,
    Compound,
    Aiger,
    Pdr,
    Kinduction,
    RandomWalk,
    Heuristic,
    BfsWitness,
    BfsCounterexample,
    ExhaustiveCompletion,
    /// Resolved by the exact symbolic Decision-Diagram reachability engine
    /// (feature `dd-backend`), which builds the complete reachable set of a
    /// small bounded net and evaluates EF/AG against it.
    Dd,
    /// Resolved by the exhaustive GPU explicit-BFS lane (feature `gpu`):
    /// formula predicates compiled to device invariants — a published
    /// violation row is a reachable witness marking; a clean completion is
    /// an exhaustive proof.
    Gpu,
}

/// Attribution record for a resolved reachability verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ReachabilityResolution {
    pub(crate) source: ReachabilityResolutionSource,
    pub(crate) depth: Option<usize>,
}

/// Prepared reachability property: classified before BFS starts.
pub(crate) enum PreparedProperty {
    /// Formula has unresolved names — emit `CANNOT_COMPUTE` without BFS.
    Invalid { id: String },
    /// Formula is valid — participate in BFS at observer slot `slot`.
    Valid { slot: usize },
}

/// Per-property tracking state during BFS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PropertyTracker {
    pub(crate) id: String,
    pub(crate) quantifier: PathQuantifier,
    pub(crate) predicate: ResolvedPredicate,
    /// Definitive verdict, if determined (may be pre-seeded by BMC).
    pub(crate) verdict: Option<bool>,
    /// Which pipeline phase resolved this property (first-writer-wins).
    pub(crate) resolved_by: Option<ReachabilityResolution>,
    /// Whether this property's FORMULA line has been printed to stdout already.
    /// Set by [`flush_resolved`] for crash-resilient incremental output.
    pub(crate) flushed: bool,
}

/// Resolve a property tracker's verdict with provenance attribution.
///
/// First-writer-wins: if the tracker already has a verdict, does nothing.
/// When `PNML_REACH_TRACE` is set and the tracker id matches a substring,
/// emits a trace line to stderr.
pub(crate) fn resolve_tracker(
    tracker: &mut PropertyTracker,
    verdict: bool,
    source: ReachabilityResolutionSource,
    depth: Option<usize>,
) {
    if tracker.verdict.is_some() {
        return;
    }
    tracker.verdict = Some(verdict);
    tracker.resolved_by = Some(ReachabilityResolution { source, depth });

    if let Ok(filter) = std::env::var("PNML_REACH_TRACE") {
        if filter.split(',').any(|sub| tracker.id.contains(sub.trim())) {
            eprintln!(
                "REACH-TRACE property={} phase={:?} verdict={} depth={}",
                tracker.id,
                source,
                if verdict { "TRUE" } else { "FALSE" },
                depth.map_or("-".to_string(), |d| d.to_string()),
            );
        }
    }
}

/// Stamp unresolved trackers as `ExhaustiveCompletion` after a completed BFS.
///
/// Call only when BFS exploration is confirmed complete (`result.completed == true`).
pub(crate) fn finalize_exhaustive_completion(trackers: &mut [PropertyTracker]) {
    for tracker in trackers.iter_mut() {
        if tracker.verdict.is_some() {
            continue;
        }
        let verdict = match tracker.quantifier {
            PathQuantifier::EF => false,
            PathQuantifier::AG => true,
        };
        resolve_tracker(
            tracker,
            verdict,
            ReachabilityResolutionSource::ExhaustiveCompletion,
            None,
        );
    }
}

fn invariant_fact_from_tracker(tracker: &PropertyTracker) -> Option<ResolvedPredicate> {
    match (tracker.quantifier, tracker.verdict) {
        (PathQuantifier::AG, Some(true)) => Some(tracker.predicate.clone()),
        (PathQuantifier::EF, Some(false)) => {
            Some(ResolvedPredicate::Not(Box::new(tracker.predicate.clone())))
        }
        _ => None,
    }
}

fn compound_verdict_from_invariants(
    invariants: &HashSet<ResolvedPredicate>,
    predicate: &ResolvedPredicate,
) -> Option<bool> {
    if invariants.contains(predicate) {
        return Some(true);
    }
    if let ResolvedPredicate::Not(inner) = predicate {
        if invariants.contains(inner.as_ref()) {
            return Some(false);
        }
    }
    let negated = ResolvedPredicate::Not(Box::new(predicate.clone()));
    invariants.contains(&negated).then_some(false)
}

/// Reuse proved reachability invariants to answer structurally related queries.
///
/// `AG(phi)=TRUE` contributes invariant fact `phi`. `EF(phi)=FALSE`
/// contributes invariant fact `NOT phi`. Pending identical predicates are true
/// on every reachable marking; pending directly-negated predicates are false on
/// every reachable marking. Petri net state spaces are non-empty because the
/// initial marking is reachable, so an always-true predicate also has an EF
/// witness and an always-false predicate also falsifies AG.
pub(in crate::examinations) fn run_compound_invariant_reuse(
    trackers: &mut [PropertyTracker],
) -> usize {
    let invariants: HashSet<_> = trackers
        .iter()
        .filter_map(invariant_fact_from_tracker)
        .collect();
    if invariants.is_empty() {
        return 0;
    }

    let mut seeded = 0;
    for tracker in trackers.iter_mut() {
        if tracker.verdict.is_some() {
            continue;
        }
        if let Some(verdict) = compound_verdict_from_invariants(&invariants, &tracker.predicate) {
            resolve_tracker(
                tracker,
                verdict,
                ReachabilityResolutionSource::Compound,
                None,
            );
            seeded += 1;
        }
    }
    seeded
}

/// Count unresolved names in a reachability formula's state predicate.
fn count_unresolved_reachability(
    formula: &ReachabilityFormula,
    aliases: &PropertyAliases,
) -> (usize, usize) {
    count_unresolved_with_aliases(&formula.predicate, aliases)
}

/// Prepare resolved trackers from properties, classifying as Valid or Invalid.
///
/// Returns `(prepared_order, trackers, validation_targets)` where trackers and
/// validation targets correspond to Valid entries at the same slot.
pub(in crate::examinations) fn prepare_trackers_with_aliases(
    original_properties: &[Property],
    simplified_properties: &[Property],
    aliases: &PropertyAliases,
) -> (
    Vec<PreparedProperty>,
    Vec<PropertyTracker>,
    Vec<WitnessValidationTarget>,
) {
    debug_assert_eq!(original_properties.len(), simplified_properties.len());

    let mut trackers = Vec::new();
    let mut validation_targets = Vec::new();
    let prepared: Vec<PreparedProperty> = original_properties
        .iter()
        .zip(simplified_properties.iter())
        .filter_map(|(original, simplified)| {
            let Formula::Reachability(ref original_rf) = original.formula else {
                return None;
            };
            let Formula::Reachability(ref simplified_rf) = simplified.formula else {
                return None;
            };
            let (original_total, original_unresolved) =
                count_unresolved_reachability(original_rf, aliases);
            let (simplified_total, simplified_unresolved) =
                count_unresolved_reachability(simplified_rf, aliases);
            if original_unresolved > 0 || simplified_unresolved > 0 {
                let total = original_total + simplified_total;
                let unresolved = original_unresolved + simplified_unresolved;
                eprintln!(
                    "Reachability resolution guard: {} has {unresolved}/{total} \
                     unresolved names → CANNOT_COMPUTE",
                    simplified.id
                );
                Some(PreparedProperty::Invalid {
                    id: simplified.id.clone(),
                })
            } else {
                let slot = trackers.len();
                let resolved = resolve_predicate_with_aliases(&simplified_rf.predicate, aliases);
                trackers.push(PropertyTracker {
                    id: simplified.id.clone(),
                    quantifier: simplified_rf.quantifier,
                    predicate: resolved,
                    verdict: None,
                    resolved_by: None,
                    flushed: false,
                });
                validation_targets.push(WitnessValidationTarget {
                    original_predicate: resolve_predicate_with_aliases(
                        &original_rf.predicate,
                        aliases,
                    ),
                });
                Some(PreparedProperty::Valid { slot })
            }
        })
        .collect();

    (prepared, trackers, validation_targets)
}

fn reachability_legacy_evidence(tracker: &PropertyTracker, completed: bool) -> EvidenceSet {
    if let Some(ReachabilityResolution {
        source:
            ReachabilityResolutionSource::BfsWitness | ReachabilityResolutionSource::BfsCounterexample,
        depth: Some(length),
    }) = tracker.resolved_by
    {
        return EvidenceSet::single(Evidence::WitnessTrace { length }, Technique::Explicit);
    }

    let lane = match tracker.resolved_by.map(|resolution| resolution.source) {
        Some(ReachabilityResolutionSource::Bmc) => "reachability-bmc",
        Some(ReachabilityResolutionSource::Lp) => "reachability-lp",
        Some(ReachabilityResolutionSource::Compound) => "reachability-compound",
        Some(ReachabilityResolutionSource::Aiger) => "reachability-aiger",
        Some(ReachabilityResolutionSource::Pdr) => "reachability-pdr",
        Some(ReachabilityResolutionSource::Kinduction) => "reachability-kinduction",
        Some(ReachabilityResolutionSource::RandomWalk) => "reachability-random-walk",
        Some(ReachabilityResolutionSource::Heuristic) => "reachability-heuristic",
        Some(ReachabilityResolutionSource::BfsWitness) => "reachability-bfs-witness",
        Some(ReachabilityResolutionSource::BfsCounterexample) => "reachability-bfs-counterexample",
        Some(ReachabilityResolutionSource::ExhaustiveCompletion) => "reachability-exhaustive",
        Some(ReachabilityResolutionSource::Dd) => "reachability-dd",
        Some(ReachabilityResolutionSource::Gpu) => "reachability-gpu",
        None if completed => "reachability-completed",
        None => "reachability",
    };
    EvidenceSet::legacy_explicit(lane)
}

fn exact_bool_record(
    id: String,
    verdict: bool,
    evidence: EvidenceSet,
) -> CertifiedExaminationRecord {
    let value = if verdict {
        BoolVerdict::True
    } else {
        BoolVerdict::False
    };
    CertifiedExaminationRecord::exact_bool(id, value, evidence)
}

fn completed_unresolved_verdict(tracker: &PropertyTracker) -> bool {
    match tracker.quantifier {
        PathQuantifier::EF => false,
        PathQuantifier::AG => true,
    }
}

fn certified_to_legacy_pair(record: CertifiedExaminationRecord) -> (String, Verdict) {
    let legacy = record.to_legacy_record();
    let verdict = match legacy.value {
        ExaminationValue::Verdict(verdict) => verdict,
        _ => Verdict::CannotCompute,
    };
    (legacy.formula_id, verdict)
}

/// Assemble certified final results from prepared classification and trackers.
///
/// This mirrors [`assemble_results`] while keeping the legacy public return
/// type as a compatibility wrapper.
pub(in crate::examinations) fn assemble_results_certified(
    prepared: &[PreparedProperty],
    trackers: &[PropertyTracker],
    completed: bool,
    skip_flushed: bool,
) -> Vec<CertifiedExaminationRecord> {
    prepared
        .iter()
        .filter_map(|prepared| match prepared {
            PreparedProperty::Invalid { id } => Some(CertifiedExaminationRecord::unknown_bool(
                id.clone(),
                UnknownReason::UnsupportedFormula,
            )),
            PreparedProperty::Valid { slot } => {
                let tracker = &trackers[*slot];
                if skip_flushed && tracker.flushed {
                    return None;
                }
                match tracker.verdict {
                    Some(verdict) => Some(exact_bool_record(
                        tracker.id.clone(),
                        verdict,
                        reachability_legacy_evidence(tracker, completed),
                    )),
                    None if completed => Some(exact_bool_record(
                        tracker.id.clone(),
                        completed_unresolved_verdict(tracker),
                        reachability_legacy_evidence(tracker, completed),
                    )),
                    None => Some(CertifiedExaminationRecord::unknown_bool(
                        tracker.id.clone(),
                        UnknownReason::IncompleteExploration {
                            visited_states: None,
                        },
                    )),
                }
            }
        })
        .collect()
}

/// Print certified FORMULA lines for newly-resolved properties.
///
/// Kept behind [`flush_resolved`] as a no-runtime-behavior compatibility step.
pub(in crate::examinations) fn flush_certified(trackers: &mut [PropertyTracker]) -> usize {
    let mut count = 0;
    for tracker in trackers.iter_mut() {
        let Some(verdict) = tracker.verdict else {
            continue;
        };
        if tracker.flushed {
            continue;
        }
        let record = exact_bool_record(
            tracker.id.clone(),
            verdict,
            reachability_legacy_evidence(tracker, false),
        );
        crate::output::print_mcc_line(record.to_mcc_line());
        tracker.flushed = true;
        count += 1;
    }
    count
}

/// Print FORMULA lines for newly-resolved properties and mark them as flushed.
///
/// Each resolved-but-unflushed tracker gets a FORMULA line printed to stdout
/// immediately, then is marked `flushed = true`. This provides crash-resilient
/// output: if the process is killed mid-pipeline, all previously-flushed
/// results survive on stdout.
///
/// Returns the number of newly flushed properties.
pub(in crate::examinations) fn flush_resolved(trackers: &mut [PropertyTracker]) -> usize {
    flush_certified(trackers)
}

/// Assemble final ordered results from prepared classification and tracker verdicts.
///
/// When `skip_flushed` is true, properties already printed by [`flush_resolved`]
/// are omitted from the returned vec to prevent duplicate FORMULA lines.
pub(in crate::examinations) fn assemble_results(
    prepared: &[PreparedProperty],
    trackers: &[PropertyTracker],
    completed: bool,
    skip_flushed: bool,
) -> Vec<(String, Verdict)> {
    assemble_results_certified(prepared, trackers, completed, skip_flushed)
        .into_iter()
        .map(certified_to_legacy_pair)
        .collect()
}

#[cfg(test)]
pub(crate) fn prepare_trackers(
    net: &crate::petri_net::PetriNet,
    properties: &[Property],
) -> (Vec<PreparedProperty>, Vec<PropertyTracker>) {
    let aliases = PropertyAliases::identity(net);
    let (prepared, trackers, _) = prepare_trackers_with_aliases(properties, properties, &aliases);
    (prepared, trackers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::petri_net::PlaceIdx;
    use crate::resolved_predicate::ResolvedIntExpr;

    fn p0_le_one() -> ResolvedPredicate {
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
            ResolvedIntExpr::Constant(1),
        )
    }

    fn tracker(
        id: &str,
        quantifier: PathQuantifier,
        predicate: ResolvedPredicate,
        verdict: Option<bool>,
    ) -> PropertyTracker {
        PropertyTracker {
            id: id.to_string(),
            quantifier,
            predicate,
            verdict,
            resolved_by: verdict.map(|_| ReachabilityResolution {
                source: ReachabilityResolutionSource::Lp,
                depth: None,
            }),
            flushed: false,
        }
    }

    #[test]
    fn bfs_witness_uses_certified_witness_trace_evidence() {
        let tracker = PropertyTracker {
            id: "prop".to_string(),
            quantifier: PathQuantifier::EF,
            predicate: ResolvedPredicate::True,
            verdict: Some(true),
            resolved_by: Some(ReachabilityResolution {
                source: ReachabilityResolutionSource::BfsWitness,
                depth: Some(3),
            }),
            flushed: false,
        };
        let records = assemble_results_certified(
            &[PreparedProperty::Valid { slot: 0 }],
            &[tracker],
            false,
            false,
        );

        assert_eq!(
            records,
            vec![CertifiedExaminationRecord::exact_bool(
                "prop",
                BoolVerdict::True,
                EvidenceSet::single(Evidence::WitnessTrace { length: 3 }, Technique::Explicit),
            )]
        );
        assert_eq!(
            records[0].to_mcc_line(),
            "FORMULA prop TRUE TECHNIQUES EXPLICIT"
        );
    }

    #[test]
    fn compound_reuse_resolves_identical_predicates_from_ag_invariant() {
        let predicate = p0_le_one();
        let mut trackers = vec![
            tracker(
                "ag-source",
                PathQuantifier::AG,
                predicate.clone(),
                Some(true),
            ),
            tracker("ef-copy", PathQuantifier::EF, predicate.clone(), None),
            tracker("ag-copy", PathQuantifier::AG, predicate, None),
        ];

        assert_eq!(run_compound_invariant_reuse(&mut trackers), 2);
        assert_eq!(trackers[1].verdict, Some(true));
        assert_eq!(trackers[2].verdict, Some(true));
        assert_eq!(
            trackers[1].resolved_by.unwrap().source,
            ReachabilityResolutionSource::Compound
        );
        assert_eq!(
            reachability_legacy_evidence(&trackers[1], false),
            EvidenceSet::legacy_explicit("reachability-compound")
        );
    }

    #[test]
    fn compound_reuse_resolves_negated_predicates_from_ag_invariant() {
        let predicate = p0_le_one();
        let negated = ResolvedPredicate::Not(Box::new(predicate.clone()));
        let mut trackers = vec![
            tracker("ag-source", PathQuantifier::AG, predicate, Some(true)),
            tracker("ef-not", PathQuantifier::EF, negated.clone(), None),
            tracker("ag-not", PathQuantifier::AG, negated, None),
        ];

        assert_eq!(run_compound_invariant_reuse(&mut trackers), 2);
        assert_eq!(trackers[1].verdict, Some(false));
        assert_eq!(trackers[2].verdict, Some(false));
        assert_eq!(
            trackers[2].resolved_by.unwrap().source,
            ReachabilityResolutionSource::Compound
        );
    }

    #[test]
    fn compound_reuse_uses_ef_false_as_negated_invariant() {
        let predicate = p0_le_one();
        let negated = ResolvedPredicate::Not(Box::new(predicate.clone()));
        let mut trackers = vec![
            tracker(
                "ef-source",
                PathQuantifier::EF,
                predicate.clone(),
                Some(false),
            ),
            tracker("ef-copy", PathQuantifier::EF, predicate, None),
            tracker("ag-not", PathQuantifier::AG, negated, None),
        ];

        assert_eq!(run_compound_invariant_reuse(&mut trackers), 2);
        assert_eq!(trackers[1].verdict, Some(false));
        assert_eq!(trackers[2].verdict, Some(true));
    }

    #[test]
    fn compound_reuse_noops_without_invariant_facts() {
        let predicate = p0_le_one();
        let mut trackers = vec![
            tracker(
                "ef-source",
                PathQuantifier::EF,
                predicate.clone(),
                Some(true),
            ),
            tracker("ag-pending", PathQuantifier::AG, predicate, None),
        ];

        assert_eq!(run_compound_invariant_reuse(&mut trackers), 0);
        assert_eq!(trackers[1].verdict, None);
        assert!(trackers[1].resolved_by.is_none());
    }
}
