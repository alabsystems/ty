// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! k-Induction for reachability properties via ay.

use std::time::Instant;

use crate::petri_net::PetriNet;
use crate::property_xml::PathQuantifier;

use super::bmc_runner::emit_kinduction_preamble;
use super::reachability::{resolve_tracker, PropertyTracker, ReachabilityResolutionSource};
use super::smt_encoding::{
    encode_predicate, find_ay, run_ay, SolverOutcome, DEPTH_LADDER, PER_DEPTH_TIMEOUT,
};

/// k-induction depth ladder.
///
/// Soundness requires the witness-side base case to cover at least the same
/// maximum depth. The `max_bmc_depth` parameter gates which depths are sound.
const KINDUCTION_DEPTH_LADDER: &[usize] = DEPTH_LADDER;
const KINDUCTION_SINGLE_PROPERTY_ENV: &str = "TY_PETRI_KINDUCTION_SINGLE_PROPERTY";

/// Run k-induction seeding on pre-resolved trackers.
///
/// For each unresolved AG property, attempts to prove it by induction:
///   UNSAT at depth k → AG(φ) = TRUE (k-inductive).
/// For each unresolved EF property, attempts to prove `AG(¬φ)` by induction:
///   UNSAT at depth k → EF(φ) = FALSE.
///
/// SAT, unknown, timeout, and process failure are all treated as
/// inconclusive (verdict left as `None`).
///
/// `max_bmc_depth`: the maximum depth at which BMC completed without UNKNOWN
/// for all pending properties. k-induction at depth k requires the base case
/// to cover at least k states, so only depths ≤ `max_bmc_depth + 1` are
/// attempted. Pass `None` to skip k-induction entirely (no base case).
pub(crate) fn run_kinduction_seeding(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    deadline: Option<Instant>,
    max_bmc_depth: Option<usize>,
) {
    let max_kind_depth = match max_bmc_depth {
        Some(d) => d + 1,
        None => return, // no base case established by BMC
    };
    let ay_path = match find_ay() {
        Some(path) => path,
        None => return,
    };
    run_kinduction_seeding_with_solver_path(net, trackers, deadline, &ay_path, max_kind_depth);
}

/// Run k-induction seeding with an explicit solver path.
///
/// `max_kind_depth`: maximum k-induction depth that has a sound base case.
/// Only depths ≤ `max_kind_depth` are attempted.
///
/// Uses a multi-property push/pop script by default. Set
/// `TY_PETRI_KINDUCTION_SINGLE_PROPERTY=1` to fall back to the legacy
/// one-process-per-property path if a solver regression appears.
///
/// SOUNDNESS: on the default batched path, every verdict-bearing UNSAT is
/// independently re-verified on a FRESH, ISOLATED ay process (no push/pop)
/// before a definite SAFE (AG=TRUE / EF=FALSE) is emitted. This closes the
/// documented ay push/pop clause-leak channel, where learned clauses from one
/// property's block could leak into a later block and produce a spurious UNSAT
/// — accepting a non-k-inductive property as SAFE when a real counterexample
/// exists. The re-check fails OPEN (verdict left unresolved for other engines)
/// rather than to a wrong answer. See [`independent_kinduction_reverify`].
fn run_kinduction_seeding_with_solver_path(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    deadline: Option<Instant>,
    ay_path: &std::path::Path,
    max_kind_depth: usize,
) {
    let unresolved: Vec<usize> = trackers
        .iter()
        .enumerate()
        .filter(|(_, tracker)| tracker.verdict.is_none())
        .map(|(index, _)| index)
        .collect();

    if unresolved.is_empty() {
        return;
    }

    let single_property_fallback = kinduction_single_property_fallback_enabled();

    for &depth in KINDUCTION_DEPTH_LADDER {
        if depth > max_kind_depth {
            break;
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            break;
        }

        let pending: Vec<usize> = unresolved
            .iter()
            .copied()
            .filter(|&index| trackers[index].verdict.is_none())
            .collect();

        if pending.is_empty() {
            break;
        }

        let mut had_unknown = false;
        let mut had_failure = false;

        if single_property_fallback {
            // Each property already runs in its own fresh ay process with no
            // push/pop, so a UNSAT here cannot be the product of cross-property
            // learned-clause carryover. The verdict-bearing UNSAT is therefore
            // its own isolated proof and needs no additional re-verification.
            for &property_idx in &pending {
                if deadline.is_some_and(|d| Instant::now() >= d) {
                    break;
                }

                let script = encode_kinduction_script_single(net, trackers, property_idx, depth);
                let timeout = deadline
                    .map(|d| PER_DEPTH_TIMEOUT.min(d.saturating_duration_since(Instant::now())))
                    .unwrap_or(PER_DEPTH_TIMEOUT);
                let result = run_ay(ay_path, &script, 1, timeout);

                match result.as_deref() {
                    Some([outcome]) => {
                        if apply_kinduction_outcome(trackers, property_idx, depth, *outcome)
                            == KinductionDepthOutcome::Unknown
                        {
                            had_unknown = true;
                        }
                    }
                    _ => had_failure = true,
                }
            }
        } else {
            let script =
                encode_kinduction_script_with_strengthening(net, trackers, &pending, depth, true);
            let timeout = deadline
                .map(|d| PER_DEPTH_TIMEOUT.min(d.saturating_duration_since(Instant::now())))
                .unwrap_or(PER_DEPTH_TIMEOUT);
            let result = run_ay(ay_path, &script, pending.len(), timeout);

            match result.as_deref() {
                Some(outcomes) if outcomes.len() == pending.len() => {
                    for (&property_idx, &outcome) in pending.iter().zip(outcomes.iter()) {
                        // SOUNDNESS GATE: the batched script shares one ay
                        // process across every property's `(push 1) ... (pop 1)`
                        // block. The documented ay push/pop defect can leak
                        // learned clauses across blocks and produce a spurious
                        // UNSAT for a later property, which would otherwise be
                        // accepted here as a definite SAFE (AG=TRUE / EF=FALSE)
                        // even when a real counterexample exists. Before any
                        // batched UNSAT is allowed to emit a verdict, re-prove
                        // that single property on a FRESH, ISOLATED ay process
                        // with no push/pop state (encode_kinduction_script_single),
                        // mirroring the IC3 independent-re-verification pattern.
                        // The re-check fails OPEN: if it does not also return
                        // UNSAT, the property is left unresolved for later
                        // engines rather than emitting a possibly-wrong SAFE.
                        let confirmed_outcome = if outcome == SolverOutcome::Unsat {
                            independent_kinduction_reverify(
                                net,
                                trackers,
                                property_idx,
                                depth,
                                ay_path,
                                deadline,
                            )
                        } else {
                            outcome
                        };

                        if apply_kinduction_outcome(
                            trackers,
                            property_idx,
                            depth,
                            confirmed_outcome,
                        ) == KinductionDepthOutcome::Unknown
                        {
                            had_unknown = true;
                        }
                    }
                }
                _ => had_failure = true,
            }
        }

        if had_unknown {
            eprintln!("k-ind depth {depth}: unknown result, stopping k-induction");
            break;
        }
        if had_failure {
            eprintln!("k-ind depth {depth}: solver failed, stopping k-induction");
            break;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KinductionDepthOutcome {
    Resolved,
    Inconclusive,
    Unknown,
}

fn apply_kinduction_outcome(
    trackers: &mut [PropertyTracker],
    property_idx: usize,
    depth: usize,
    outcome: SolverOutcome,
) -> KinductionDepthOutcome {
    match outcome {
        SolverOutcome::Unsat => match trackers[property_idx].quantifier {
            PathQuantifier::AG => {
                resolve_tracker(
                    &mut trackers[property_idx],
                    true,
                    ReachabilityResolutionSource::Kinduction,
                    Some(depth),
                );
                eprintln!(
                    "k-ind depth {depth}: {} = TRUE (AG {}-inductive)",
                    trackers[property_idx].id, depth
                );
                KinductionDepthOutcome::Resolved
            }
            PathQuantifier::EF => {
                resolve_tracker(
                    &mut trackers[property_idx],
                    false,
                    ReachabilityResolutionSource::Kinduction,
                    Some(depth),
                );
                eprintln!(
                    "k-ind depth {depth}: {} = FALSE (EF, negation {}-inductive)",
                    trackers[property_idx].id, depth
                );
                KinductionDepthOutcome::Resolved
            }
        },
        SolverOutcome::Sat => KinductionDepthOutcome::Inconclusive,
        SolverOutcome::Unknown => KinductionDepthOutcome::Unknown,
    }
}

/// Independently re-verify a batched k-induction UNSAT on a FRESH, ISOLATED
/// ay process before it is allowed to emit a definite SAFE verdict.
///
/// The default verdict-emitting path batches every property's
/// `(push 1) ... (check-sat) ... (pop 1)` block into one persistent ay process.
/// The documented ay push/pop defect can leak learned clauses from one block
/// into a later block and report a *spurious* UNSAT — accepting a property that
/// is NOT genuinely k-inductive as AG=TRUE / EF=FALSE even when a real
/// counterexample exists. That is a catastrophic wrong MCC verdict.
///
/// This re-runs the single property via [`encode_kinduction_script_single`],
/// which carries NO push/pop and runs in its OWN fresh ay process with no
/// shared learned-clause state, so cross-property clause carryover cannot occur.
/// A genuinely k-inductive property still yields UNSAT here and the verdict is
/// emitted; a leaked-clause UNSAT that does not reflect a real proof yields
/// SAT/Unknown/failure on the isolated re-check, and we fail OPEN — returning a
/// non-UNSAT outcome so the property is left unresolved for later engines rather
/// than emitting a possibly-wrong SAFE.
///
/// Mirrors the IC3 independent-re-verification discipline (commit 12770074):
/// a SAFE verdict must be backed by a proof on an isolated solver, never by a
/// shared-state UNSAT.
fn independent_kinduction_reverify(
    net: &PetriNet,
    trackers: &[PropertyTracker],
    property_idx: usize,
    depth: usize,
    ay_path: &std::path::Path,
    deadline: Option<Instant>,
) -> SolverOutcome {
    if deadline.is_some_and(|d| Instant::now() >= d) {
        // No time budget for an independent proof-back; fail OPEN. Treating this
        // as Inconclusive (not Unknown) leaves the verdict None without halting
        // the depth ladder for the other properties at this depth.
        return SolverOutcome::Sat;
    }

    let script = encode_kinduction_script_single(net, trackers, property_idx, depth);
    let timeout = deadline
        .map(|d| PER_DEPTH_TIMEOUT.min(d.saturating_duration_since(Instant::now())))
        .unwrap_or(PER_DEPTH_TIMEOUT);
    let result = run_ay(ay_path, &script, 1, timeout);

    match result.as_deref() {
        // Confirmed UNSAT on an isolated solver: the k-inductive proof is
        // genuine, so the batched verdict may be emitted.
        Some([SolverOutcome::Unsat]) => SolverOutcome::Unsat,
        // The isolated re-check did NOT confirm the batched UNSAT (it found SAT,
        // returned Unknown, or the process failed). Fail OPEN to other lanes:
        // report Inconclusive (Sat) so no SAFE verdict is emitted and the depth
        // ladder is not prematurely halted for this depth.
        _ => SolverOutcome::Sat,
    }
}

fn kinduction_single_property_fallback_enabled() -> bool {
    std::env::var(KINDUCTION_SINGLE_PROPERTY_ENV)
        .ok()
        .as_deref()
        .is_some_and(is_truthy_env_value)
}

fn is_truthy_env_value(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Generate SMT-LIB script for k-induction of a single property at a given depth.
///
/// No push/pop — one complete script per property, run in its own ay process.
/// This avoids the ay push/pop soundness bug where learned clauses from earlier
/// blocks can produce incorrect UNSAT for later properties.
fn encode_kinduction_script_single(
    net: &PetriNet,
    trackers: &[PropertyTracker],
    property_idx: usize,
    depth: usize,
) -> String {
    let mut script = String::with_capacity(4096);

    emit_kinduction_preamble(&mut script, net, depth);

    let tracker = &trackers[property_idx];
    let hypothesis_negated = tracker.quantifier == PathQuantifier::EF;
    for step in 0..depth {
        let predicate = encode_predicate(&tracker.predicate, step, net);
        if hypothesis_negated {
            script.push_str(&format!("(assert (not {}))\n", predicate));
        } else {
            script.push_str(&format!("(assert {})\n", predicate));
        }
    }

    let predicate_at_depth = encode_predicate(&tracker.predicate, depth, net);
    if hypothesis_negated {
        script.push_str(&format!("(assert {})\n", predicate_at_depth));
    } else {
        script.push_str(&format!("(assert (not {}))\n", predicate_at_depth));
    }

    script.push_str("(check-sat)\n");
    script.push_str("(exit)\n");
    script
}

/// Generate SMT-LIB script for k-induction at a given depth (multi-property, push/pop).
///
/// Used by the production path by default. The single-property encoder remains
/// available as an explicit fallback if a solver push/pop regression appears.
#[cfg(test)]
fn encode_kinduction_script(
    net: &PetriNet,
    trackers: &[PropertyTracker],
    pending: &[usize],
    depth: usize,
) -> String {
    encode_kinduction_script_with_strengthening(net, trackers, pending, depth, true)
}

/// Generate SMT-LIB script for k-induction at a given depth.
///
/// Key differences from BMC:
/// - No initial marking constraints (arbitrary start state)
/// - Per-property: assert the property at steps 0..k-1 (induction hypothesis)
/// - Per-property: assert the negation at step k (induction check)
/// - For EF properties: induction is on `¬φ` (proving AG(¬φ))
///
/// When `strengthen_step0` is true, adds state-equation constraints at step 0:
///   M0_model = M0_real + C * parikh  (parikh >= 0)
/// This restricts step-0 markings to those reachable via the state equation,
/// pruning spurious states that would make induction inconclusive.
fn encode_kinduction_script_with_strengthening(
    net: &PetriNet,
    trackers: &[PropertyTracker],
    pending: &[usize],
    depth: usize,
    strengthen_step0: bool,
) -> String {
    use super::bmc_runner::{emit_marking_vars, emit_non_negativity, emit_step_vars};
    use super::smt_encoding::encode_transition_relation;

    let num_places = net.num_places();
    let num_transitions = net.num_transitions();
    let mut script = String::with_capacity(4096);

    if strengthen_step0 {
        emit_kinduction_preamble(&mut script, net, depth);
    } else {
        // Bare preamble without state-equation or P-invariant strengthening.
        script.push_str("(set-logic QF_LIA)\n");
        emit_marking_vars(&mut script, num_places, depth);
        emit_step_vars(&mut script, num_transitions, depth);
        emit_non_negativity(&mut script, num_places, depth);
        encode_transition_relation(&mut script, net, num_places, num_transitions, depth);
    }

    for &property_idx in pending {
        let tracker = &trackers[property_idx];
        script.push_str("(push 1)\n");

        let hypothesis_negated = tracker.quantifier == PathQuantifier::EF;
        for step in 0..depth {
            let predicate = encode_predicate(&tracker.predicate, step, net);
            if hypothesis_negated {
                script.push_str(&format!("(assert (not {}))\n", predicate));
            } else {
                script.push_str(&format!("(assert {})\n", predicate));
            }
        }

        let predicate_at_depth = encode_predicate(&tracker.predicate, depth, net);
        if hypothesis_negated {
            script.push_str(&format!("(assert {})\n", predicate_at_depth));
        } else {
            script.push_str(&format!("(assert (not {}))\n", predicate_at_depth));
        }

        script.push_str("(check-sat)\n");
        script.push_str("(pop 1)\n");
    }

    script.push_str("(exit)\n");
    script
}

#[cfg(test)]
#[path = "kinduction_tests.rs"]
mod tests;
