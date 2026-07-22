// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! AIGER-based reachability seeding via Petri-to-AIGER cross-encoding.
//!
//! For bounded Petri nets, encodes the net as an AIGER circuit and runs the
//! tla-aiger IC3/BMC portfolio. This can resolve both AG and EF properties:
//!
//! - `AG(phi)`: encode `NOT phi` as the bad-state output. Safe => AG=TRUE,
//!   Unsafe => AG=FALSE.
//! - `EF(phi)`: encode `phi` as the bad-state output (negate the safety
//!   encoding). Safe => EF=FALSE, Unsafe => EF=TRUE.
//!
//! The cross-encoding only fires when:
//! 1. All places have finite LP upper bounds
//! 2. Total latch count <= 500 (the `encode_aiger` gating threshold)
//! 3. The property is in the supported predicate subset: marking predicates
//!    plus IsFireable over known transitions
//!
//! This phase runs after LP seeding (which computes the bounds we need) and
//! before PDR seeding (AIGER portfolio is stronger for bounded nets).

use std::collections::HashMap;
use std::time::{Duration, Instant};

#[allow(unused_imports)]
use crate::encode_aiger::{try_encode_as_aiger, try_encode_as_aiger_with_nupn, AigerTraceMap};
use crate::intelligence_bus::IntelligenceBus;
use crate::lp_state_equation::lp_upper_bound;
use crate::nupn::NupnStructure;
use crate::petri_net::{PetriNet, PlaceIdx, TransitionIdx};
use crate::property_xml::PathQuantifier;
use crate::resolved_predicate::ResolvedPredicate;

use super::reachability::{resolve_tracker, PropertyTracker, ReachabilityResolutionSource};
use super::reachability_witness::{
    apply_validated_witnesses, WitnessCandidate, WitnessSeedSource, WitnessValidationContext,
};

/// Hard cap for one AIGER portfolio dispatch.
const AIGER_SEED_TIMEOUT: Duration = Duration::from_secs(10);

fn fair_share_duration(remaining: Duration, pending_count: usize) -> Duration {
    let divisor = pending_count.clamp(1, u32::MAX as usize) as u32;
    remaining / divisor
}

fn aiger_property_deadline_at(
    global_deadline: Option<Instant>,
    pending_count: usize,
    now: Instant,
) -> Option<Instant> {
    global_deadline.map(|deadline| {
        let remaining = deadline.saturating_duration_since(now);
        let budget = AIGER_SEED_TIMEOUT.min(fair_share_duration(remaining, pending_count));
        now + budget
    })
}

/// Unified verdict from a single AIGER property check.
///
/// Produced by [`run_aiger_property`]. The caller is responsible for
/// interpreting the verdict in the context of their specific examination:
///
/// - `witness = Some(trace)`: the AIGER portfolio returned SAT and the trace
///   was successfully replayed as a concrete Petri transition sequence. For
///   `AG(phi)` queries, this is a counterexample to `phi`. For deadlock
///   queries (safety = `IsFireable(all)`), this is a witness path to a
///   marking in which no transition is enabled. Callers MUST perform any
///   additional terminal-state checks they require (e.g., deadlock callers
///   re-verify that no transition is enabled at the end of the replayed
///   trace).
/// - `unsat = true`: the AIGER portfolio proved the bad output unreachable
///   on the circuit. CALLER must decide whether to trust this for their
///   specific property — for properties that contain `IsFireable` terms,
///   UNSAT is unsound and must be ignored (see
///   [`predicate_contains_fireability`]).
/// - Both `witness = None` and `unsat = false`: encoding rejected,
///   timeout/Unknown from the portfolio, or replay failed.
pub(crate) struct AigerPropertyVerdict {
    /// SAT witness as a Petri trace, if the property's bad-state was
    /// reachable AND the AIGER trace replayed successfully.
    pub witness: Option<Vec<TransitionIdx>>,
    /// `true` iff the AIGER portfolio returned UNSAT on the circuit.
    pub unsat: bool,
}

/// Unified AIGER property check: encode + run portfolio + replay-on-SAT.
///
/// This is the single entry point every AIGER-backed examination should use.
/// Any improvement here (better encoding, smarter portfolio config,
/// additional preprocessing, etc.) benefits every caller automatically.
///
/// Returns `None` only when AIGER cannot be applied (unbounded net, encoding
/// outside the supported subset, latch cap exceeded, or zero remaining
/// timeout). Returns `Some(AigerPropertyVerdict)` otherwise; the caller
/// inspects `witness` / `unsat` per the soundness rules documented on
/// [`AigerPropertyVerdict`].
pub(crate) fn run_aiger_property(
    net: &PetriNet,
    safety_property: &ResolvedPredicate,
    deadline: Option<Instant>,
    nupn: Option<&NupnStructure>,
    bus: Option<&IntelligenceBus>,
) -> Option<AigerPropertyVerdict> {
    // 1. LP (and optional NUPN one-safe) bounds for every place.
    // The intelligence bus, if provided, supplies cached per-place LP bounds
    // seeded once upstream — avoids re-solving N independent LPs per call.
    let bounds = compute_all_place_bounds_with_nupn(net, nupn, bus)?;

    // 2. Compute timeout: min(AIGER_SEED_TIMEOUT, deadline.remaining()).
    let timeout = deadline
        .map(|limit| AIGER_SEED_TIMEOUT.min(limit.saturating_duration_since(Instant::now())))
        .unwrap_or(AIGER_SEED_TIMEOUT);
    if timeout.is_zero() {
        return None;
    }

    // 3. Try AIGER encoding. None => outside subset / over latch cap / etc.
    let encoding = try_encode_as_aiger_with_nupn(net, safety_property, &bounds, nupn)?;

    eprintln!(
        "AIGER encoding: {} latches, {} transitions, {} constraints",
        encoding.num_latches,
        encoding.num_transitions,
        encoding.circuit.constraints.len(),
    );

    // 4. Run the tla-aiger portfolio.
    let results = tla_aiger::check_aiger_sat(&encoding.circuit, Some(timeout));
    let Some(result) = results.first() else {
        return Some(AigerPropertyVerdict {
            witness: None,
            unsat: false,
        });
    };

    match result {
        tla_aiger::AigerCheckResult::Unsat => Some(AigerPropertyVerdict {
            witness: None,
            unsat: true,
        }),
        tla_aiger::AigerCheckResult::Sat { trace } => {
            // Replay the AIGER trace into a concrete Petri trace. On replay
            // failure, surface that as "no witness" so the caller can decide
            // how to proceed (it is NOT the same as UNSAT).
            let replay = match aiger_trace_to_petri_trace(net, &encoding.trace_map, trace) {
                Ok(replay) => replay,
                Err(reason) => {
                    eprintln!("AIGER witness rejected (replay failed): {reason}");
                    return Some(AigerPropertyVerdict {
                        witness: None,
                        unsat: false,
                    });
                }
            };

            if !replay.diagnostics.latch_mismatches.is_empty() {
                eprintln!(
                    "AIGER witness latch cross-check: {} mismatch(es), \
                     {} checked frame(s), {} incomplete frame(s)",
                    replay.diagnostics.latch_mismatches.len(),
                    replay.diagnostics.latch_frames_checked,
                    replay.diagnostics.latch_frames_incomplete,
                );
                for mismatch in replay.diagnostics.latch_mismatches.iter().take(3) {
                    eprintln!("  {mismatch}");
                }
            }

            Some(AigerPropertyVerdict {
                witness: Some(replay.petri_trace),
                unsat: false,
            })
        }
        tla_aiger::AigerCheckResult::Unknown { .. } => Some(AigerPropertyVerdict {
            witness: None,
            unsat: false,
        }),
    }
}

/// Run AIGER-based seeding on unresolved reachability trackers.
///
/// For each unresolved tracker, attempts to:
/// 1. Compute LP upper bounds for all places
/// 2. Encode the net + property as an AIGER circuit
/// 3. Run the tla-aiger portfolio on the circuit
/// 4. Translate the result back to a Petri reachability verdict
pub(crate) fn run_aiger_seeding(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    validation: &WitnessValidationContext<'_>,
    deadline: Option<Instant>,
) {
    run_aiger_seeding_with_nupn(net, trackers, validation, deadline, None, None);
}

pub(crate) fn run_aiger_seeding_with_nupn(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    validation: &WitnessValidationContext<'_>,
    deadline: Option<Instant>,
    nupn: Option<&NupnStructure>,
    bus: Option<&IntelligenceBus>,
) {
    // Eagerly reject unbounded nets so we don't pay the per-tracker setup cost.
    if compute_all_place_bounds_with_nupn(net, nupn, bus).is_none() {
        return;
    }

    for slot in 0..trackers.len() {
        if trackers[slot].verdict.is_some() {
            continue;
        }

        let property_id = trackers[slot].id.clone();
        let quantifier = trackers[slot].quantifier;
        let predicate = trackers[slot].predicate.clone();

        // For AG(phi): safety property is phi (bad = NOT phi).
        // For EF(phi): safety property is NOT phi (bad = phi).
        //   If SAFE => NOT phi always holds => phi never holds => EF=FALSE.
        //   If UNSAFE => phi is reachable => EF=TRUE.
        let safety_property = match quantifier {
            PathQuantifier::AG => predicate,
            PathQuantifier::EF => ResolvedPredicate::Not(Box::new(predicate)),
        };

        let pending_count = trackers
            .iter()
            .skip(slot)
            .filter(|tracker| tracker.verdict.is_none())
            .count();
        let property_deadline = aiger_property_deadline_at(deadline, pending_count, Instant::now());

        // Delegate the encode + portfolio + replay pipeline to the unified
        // engine. None means AIGER was not applicable (out-of-subset,
        // exhausted timeout, etc.); skip and let later phases handle it.
        let Some(verdict) = run_aiger_property(net, &safety_property, property_deadline, nupn, bus)
        else {
            continue;
        };

        if let Some(trace) = verdict.witness {
            // Safety violated at the circuit level. Resolve the Petri property
            // only if Petri replay validation accepts the witness.
            let accepted = apply_validated_witnesses(
                validation,
                trackers,
                [WitnessCandidate {
                    tracker_slot: slot,
                    source: WitnessSeedSource::Aiger,
                    trace,
                }],
            );
            if accepted == 0 {
                eprintln!("AIGER witness rejected by Petri replay for {property_id}");
            }
        } else if verdict.unsat {
            resolve_aiger_unsat(&mut trackers[slot]);
        }
        // else: Unknown/timeout/replay-fail — leave the tracker unresolved and
        // let the next phase try.
    }
}

/// Run AIGER+IC3 portfolio for a deadlock query (TRUE-only soundness gate).
///
/// Thin wrapper around the unified [`run_aiger_property`] engine. Builds the
/// safety property `IsFireable(all transitions)` (so the circuit's bad-state
/// output is "no transition is fireable" = deadlock) and validates any SAT
/// witness terminates in a true deadlock marking before returning it.
///
/// Soundness rule: AIGER UNSAT MUST NOT be turned into `Verdict::False`. The
/// safety property here contains fireability terms and the encoder
/// over-approximates fireability semantics, so circuit-UNSAT does not soundly
/// imply UNSAT on the Petri net. This wrapper drops the `unsat` branch on the
/// floor, mirroring the analogous refusal in [`resolve_aiger_unsat`] /
/// [`predicate_contains_fireability`] on the reachability path.
pub(crate) fn run_aiger_deadlock_check(
    net: &PetriNet,
    deadline: Option<Instant>,
) -> Option<Vec<TransitionIdx>> {
    let nt = net.num_transitions();
    if nt == 0 {
        // No transitions at all: the initial state is trivially a deadlock,
        // but with an empty transition vector. Let downstream phases handle
        // this edge case so the AIGER path remains a pure seeding step.
        return None;
    }

    // Safety property: IsFireable(all transitions). Bad-state = deadlock.
    let all_transitions: Vec<TransitionIdx> = (0..nt).map(|t| TransitionIdx(t as u32)).collect();
    let safety_property = ResolvedPredicate::IsFireable(all_transitions);

    // Delegate encoding + portfolio + replay to the unified engine. We only
    // act on SAT (witness); UNSAT is intentionally discarded for soundness.
    let verdict = run_aiger_property(net, &safety_property, deadline, None, None)?;
    let trace = verdict.witness?;

    // Re-validate the terminal marking is a true deadlock in the original
    // Petri net. The replay layer already filters transitions to enabled
    // ones, but we still verify the final marking has zero enabled
    // transitions before claiming a deadlock witness.
    let mut marking = net.initial_marking.clone();
    for &t in &trace {
        if !net.is_enabled(&marking, t) {
            eprintln!(
                "AIGER deadlock witness rejected: transition {} not enabled during replay",
                t.0,
            );
            return None;
        }
        // Fail-closed (#22): token-count overflow during replay means the
        // witness marking is not representable — reject the witness.
        if net.apply_delta(&mut marking, t).is_err() {
            eprintln!(
                "AIGER deadlock witness rejected: transition {} overflows place token count",
                t.0,
            );
            return None;
        }
    }

    let has_enabled = (0..nt).any(|t| net.is_enabled(&marking, TransitionIdx(t as u32)));
    if has_enabled {
        eprintln!(
            "AIGER deadlock witness rejected: terminal marking still has enabled transition(s)"
        );
        return None;
    }

    eprintln!(
        "AIGER deadlock witness accepted: {} step(s) to deadlock",
        trace.len(),
    );
    Some(trace)
}

fn resolve_aiger_unsat(tracker: &mut PropertyTracker) -> bool {
    if predicate_contains_fireability(&tracker.predicate) {
        eprintln!(
            "AIGER UNSAT left unresolved for fireability property {}; \
             awaiting exact fallback",
            tracker.id,
        );
        return false;
    }

    // Safety holds: bad state unreachable.
    let verdict = match tracker.quantifier {
        // AG(phi) + Safe => phi always holds => TRUE.
        PathQuantifier::AG => true,
        // EF(phi) encoded as safety of NOT phi. Safe => NOT phi always holds
        // => phi never reachable => EF=FALSE.
        PathQuantifier::EF => false,
    };
    resolve_tracker(tracker, verdict, ReachabilityResolutionSource::Aiger, None);
    true
}

fn predicate_contains_fireability(predicate: &ResolvedPredicate) -> bool {
    match predicate {
        ResolvedPredicate::And(children) | ResolvedPredicate::Or(children) => {
            children.iter().any(predicate_contains_fireability)
        }
        ResolvedPredicate::Not(inner) => predicate_contains_fireability(inner),
        ResolvedPredicate::IsFireable(_) => true,
        ResolvedPredicate::IntLe(..) | ResolvedPredicate::True | ResolvedPredicate::False => false,
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct AigerReplayDiagnostics {
    pub(crate) latch_frames_checked: usize,
    pub(crate) latch_frames_incomplete: usize,
    pub(crate) latch_mismatches: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AigerReplay {
    pub(crate) petri_trace: Vec<TransitionIdx>,
    pub(crate) diagnostics: AigerReplayDiagnostics,
}

pub(crate) fn aiger_trace_to_petri_trace(
    net: &PetriNet,
    trace_map: &AigerTraceMap,
    trace: &[rustc_hash::FxHashMap<String, i64>],
) -> Result<AigerReplay, String> {
    if trace.is_empty() {
        return Err("empty AIGER trace".to_string());
    }
    if trace_map.transitions.len() != net.num_transitions() {
        return Err(format!(
            "trace map has {} transitions for net with {} transitions",
            trace_map.transitions.len(),
            net.num_transitions(),
        ));
    }

    let mut marking = net.initial_marking.clone();
    let mut petri_trace = Vec::new();
    let mut diagnostics = AigerReplayDiagnostics::default();
    let check_named_latches = trace_contains_named_latches(trace_map, trace);
    if check_named_latches {
        update_latch_marking_diagnostics(&mut diagnostics, trace_map, &trace[0], &marking, 0);
    }

    for (step, model_step) in trace.iter().take(trace.len().saturating_sub(1)).enumerate() {
        let mut selected = Vec::with_capacity(net.num_transitions());
        for input in &trace_map.transitions {
            let input_selected = aiger_trace_input_bool(model_step, &input.name, input.input_index)
                .map_err(|reason| format!("step {step}: {reason}"))?;
            selected.push((input.transition, input_selected));
        }

        let mut fired = None;
        for (transition_idx, input_selected) in selected {
            if !input_selected {
                continue;
            }

            if net.is_enabled(&marking, transition_idx) {
                fired = Some(transition_idx);
                break;
            }
        }

        if let Some(transition) = fired {
            // Fail-closed (#22): a token-count overflow makes the replayed
            // marking non-representable — reject the trace rather than wrap.
            net.apply_delta(&mut marking, transition).map_err(|e| {
                format!(
                    "step {step}: transition {} overflows place token count: {e}",
                    transition.0
                )
            })?;
            petri_trace.push(transition);
        }
        if check_named_latches {
            update_latch_marking_diagnostics(
                &mut diagnostics,
                trace_map,
                &trace[step + 1],
                &marking,
                step + 1,
            );
        }
    }

    Ok(AigerReplay {
        petri_trace,
        diagnostics,
    })
}

fn aiger_trace_input_bool(
    model_step: &rustc_hash::FxHashMap<String, i64>,
    name: &str,
    input_index: usize,
) -> Result<bool, String> {
    match model_step.get(name) {
        Some(value) => aiger_trace_bool_value(name, *value),
        None => {
            let key = format!("i{input_index}");
            aiger_trace_bool(model_step, &key)
        }
    }
}

fn aiger_trace_bool(
    model_step: &rustc_hash::FxHashMap<String, i64>,
    key: &str,
) -> Result<bool, String> {
    match model_step.get(key) {
        Some(value) => aiger_trace_bool_value(key, *value),
        None => Err(format!("missing {key} in AIGER trace")),
    }
}

fn aiger_trace_bool_value(key: &str, value: i64) -> Result<bool, String> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        value => Err(format!("{key} has non-Boolean value {value}")),
    }
}

fn trace_contains_named_latches(
    trace_map: &AigerTraceMap,
    trace: &[rustc_hash::FxHashMap<String, i64>],
) -> bool {
    for place in &trace_map.places {
        if place.latch_indices.is_empty() {
            continue;
        }
        let key = format!("petri.place.{}.bit.0", place.place.0);
        if trace.iter().any(|step| step.contains_key(&key)) {
            return true;
        }
    }

    trace_map.nupn_units.iter().any(|unit| {
        let key = format!("petri.nupn.unit.{}.state.bit.0", unit.unit_index);
        trace.iter().any(|step| step.contains_key(&key))
    })
}

fn update_latch_marking_diagnostics(
    diagnostics: &mut AigerReplayDiagnostics,
    trace_map: &AigerTraceMap,
    model_step: &rustc_hash::FxHashMap<String, i64>,
    marking: &[u64],
    frame: usize,
) {
    if trace_map.places.len() != marking.len() {
        diagnostics.latch_frames_incomplete += 1;
        return;
    }

    let mut decoded = vec![None; marking.len()];
    for place in &trace_map.places {
        if place.latch_indices.is_empty() {
            continue;
        }
        let place_index = place.place.0 as usize;
        if place_index >= marking.len() {
            diagnostics.latch_frames_incomplete += 1;
            return;
        }

        let mut tokens = 0u64;
        for bit in 0..place.latch_indices.len() {
            let key = format!("petri.place.{}.bit.{bit}", place.place.0);
            let Some(raw_value) = model_step.get(&key) else {
                diagnostics.latch_frames_incomplete += 1;
                return;
            };
            let bit_value = match aiger_trace_bool_value(&key, *raw_value) {
                Ok(value) => value,
                Err(reason) => {
                    diagnostics
                        .latch_mismatches
                        .push(format!("frame {frame}: {reason}"));
                    return;
                }
            };
            if bit_value {
                tokens |= 1u64 << bit;
            }
        }
        decoded[place_index] = Some(tokens);
    }

    for unit in &trace_map.nupn_units {
        let mut code = 0u64;
        for bit in 0..unit.latch_indices.len() {
            let key = format!("petri.nupn.unit.{}.state.bit.{bit}", unit.unit_index);
            let Some(raw_value) = model_step.get(&key) else {
                diagnostics.latch_frames_incomplete += 1;
                return;
            };
            let bit_value = match aiger_trace_bool_value(&key, *raw_value) {
                Ok(value) => value,
                Err(reason) => {
                    diagnostics
                        .latch_mismatches
                        .push(format!("frame {frame}: {reason}"));
                    return;
                }
            };
            if bit_value {
                code |= 1u64 << bit;
            }
        }

        if code as usize > unit.places.len() {
            diagnostics.latch_mismatches.push(format!(
                "frame {frame}: NUPN unit {} latch code {code} exceeds {} places",
                unit.unit_index,
                unit.places.len(),
            ));
            return;
        }

        for (idx, place) in unit.places.iter().enumerate() {
            let place_index = place.0 as usize;
            if place_index >= marking.len() {
                diagnostics.latch_frames_incomplete += 1;
                return;
            }
            decoded[place_index] = Some(u64::from(code == idx as u64 + 1));
        }
    }

    diagnostics.latch_frames_checked += 1;
    for (place, actual) in decoded.into_iter().enumerate() {
        let Some(actual) = actual else {
            diagnostics.latch_frames_incomplete += 1;
            return;
        };
        let expected = marking[place];
        if actual != expected {
            diagnostics.latch_mismatches.push(format!(
                "frame {frame}: place {place} latch marking {actual} != Petri marking {expected}",
            ));
        }
    }
}

/// Compute LP upper bounds for every place in the net.
///
/// Returns `None` if any place has an unbounded token count (LP returns
/// `None` for that place), meaning the AIGER encoding is not applicable.
fn compute_all_place_bounds(net: &PetriNet) -> Option<HashMap<PlaceIdx, u64>> {
    compute_all_place_bounds_with_nupn(net, None, None)
}

/// Compute LP upper bounds for every place in the net, with optional
/// NUPN one-safe fallback and optional `IntelligenceBus` bound cache.
///
/// When `bus` is `Some`, per-place LP bounds are read from the bus cache
/// (populated upstream by [`crate::intelligence_bus::seed_from_lp`]). The
/// cache lookup is bit-exact equivalent to a fresh `lp_upper_bound` call
/// because the seed step uses the same underlying solver on the same net.
/// Bus-miss falls back to a fresh `lp_upper_bound` solve — soundness is
/// preserved regardless of cache state.
pub(crate) fn compute_all_place_bounds_with_nupn(
    net: &PetriNet,
    nupn: Option<&NupnStructure>,
    bus: Option<&IntelligenceBus>,
) -> Option<HashMap<PlaceIdx, u64>> {
    let nupn_bounds = nupn.and_then(|nupn| {
        (nupn.unit_safe() && nupn.initial_marking_respects_unit_safety(&net.initial_marking))
            .then(|| nupn.one_safe_place_bounds(net.num_places()))
    });

    let mut bounds = HashMap::with_capacity(net.num_places());
    for p in 0..net.num_places() {
        let place_idx = PlaceIdx(p as u32);
        let lp_bound = bus
            .and_then(|b| b.get_place_bound(place_idx))
            .map(Some)
            .unwrap_or_else(|| lp_upper_bound(net, &[place_idx]));
        let nupn_bound = nupn_bounds
            .as_ref()
            .and_then(|place_bounds| place_bounds[p]);
        let bound = match (lp_bound, nupn_bound) {
            (Some(lp), Some(nupn)) => lp.min(nupn),
            (Some(lp), None) => lp,
            (None, Some(nupn)) => nupn,
            (None, None) => return None,
        };
        bounds.insert(place_idx, bound);
    }
    Some(bounds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examinations::reachability::{PropertyTracker, ReachabilityResolutionSource};
    use crate::examinations::reachability_witness::{
        validation_targets_from_trackers, WitnessValidationContext,
    };
    use crate::petri_net::{Arc, PetriNet, PlaceInfo, TransitionInfo};
    use crate::property_xml::PathQuantifier;
    use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};
    use rustc_hash::FxHashMap;

    fn place(id: &str) -> PlaceInfo {
        PlaceInfo {
            id: id.to_string(),
            name: None,
        }
    }

    fn arc(place: u32, weight: u64) -> Arc {
        Arc {
            place: PlaceIdx(place),
            weight,
        }
    }

    fn trans(id: &str, inputs: Vec<Arc>, outputs: Vec<Arc>) -> TransitionInfo {
        TransitionInfo {
            id: id.to_string(),
            name: None,
            inputs,
            outputs,
        }
    }

    fn make_tracker(
        id: &str,
        quantifier: PathQuantifier,
        predicate: ResolvedPredicate,
    ) -> PropertyTracker {
        PropertyTracker {
            id: id.to_string(),
            quantifier,
            predicate,
            verdict: None,
            resolved_by: None,
            flushed: false,
        }
    }

    fn run_aiger_test(net: &PetriNet, trackers: &mut [PropertyTracker]) {
        let targets = validation_targets_from_trackers(trackers);
        let validation = WitnessValidationContext::new(net, &targets);
        run_aiger_seeding(net, trackers, &validation, None);
    }

    #[test]
    fn test_aiger_property_deadline_fair_shares_pending_trackers() {
        let now = Instant::now();
        let deadline = now + Duration::from_secs(9);

        let property_deadline = aiger_property_deadline_at(Some(deadline), 3, now)
            .expect("finite global deadline should produce a property deadline");

        assert_eq!(
            property_deadline.duration_since(now),
            Duration::from_secs(3)
        );
    }

    #[test]
    fn test_aiger_property_deadline_caps_large_share() {
        let now = Instant::now();
        let deadline = now + Duration::from_secs(120);

        let property_deadline = aiger_property_deadline_at(Some(deadline), 3, now)
            .expect("finite global deadline should produce a property deadline");

        assert_eq!(property_deadline.duration_since(now), AIGER_SEED_TIMEOUT);
    }

    fn trace_step(num_transitions: usize, requested: &[usize]) -> FxHashMap<String, i64> {
        let mut step = FxHashMap::default();
        for transition in 0..num_transitions {
            step.insert(
                format!("i{transition}"),
                i64::from(requested.contains(&transition)),
            );
        }
        step
    }

    fn insert_named_latches(
        step: &mut FxHashMap<String, i64>,
        trace_map: &AigerTraceMap,
        marking: &[u64],
    ) {
        for place in &trace_map.places {
            let tokens = marking[place.place.0 as usize];
            for bit in 0..place.latch_indices.len() {
                step.insert(
                    format!("petri.place.{}.bit.{bit}", place.place.0),
                    ((tokens >> bit) & 1) as i64,
                );
            }
        }
        for unit in &trace_map.nupn_units {
            let mut code = 0u64;
            for (idx, place) in unit.places.iter().enumerate() {
                if marking[place.0 as usize] > 0 {
                    code = idx as u64 + 1;
                    break;
                }
            }
            for bit in 0..unit.latch_indices.len() {
                step.insert(
                    format!("petri.nupn.unit.{}.state.bit.{bit}", unit.unit_index),
                    ((code >> bit) & 1) as i64,
                );
            }
        }
    }

    fn replay_trace_map(net: &PetriNet) -> AigerTraceMap {
        let bounds: HashMap<PlaceIdx, u64> = (0..net.num_places())
            .map(|place| (PlaceIdx(place as u32), net.initial_marking[place].max(1)))
            .collect();
        try_encode_as_aiger(net, &ResolvedPredicate::True, &bounds)
            .expect("test net should encode")
            .trace_map
    }

    fn one_place_nupn(net: &PetriNet) -> crate::nupn::NupnStructure {
        let pnml = r#"<?xml version="1.0"?>
<pnml>
  <net id="one-place">
    <page id="page0">
      <toolspecific tool="nupn" version="1.1">
        <structure units="1" root="u0" safe="true">
          <unit id="u0">
            <places>p0</places>
            <subunits/>
          </unit>
        </structure>
      </toolspecific>
    </page>
  </net>
</pnml>"#;
        crate::nupn::parse_nupn(pnml, net)
            .expect("NUPN should parse")
            .expect("NUPN should be present")
    }

    fn mutex_net() -> PetriNet {
        PetriNet {
            name: None,
            places: vec![place("free"), place("busy")],
            transitions: vec![
                trans("acquire", vec![arc(0, 1)], vec![arc(1, 1)]),
                trans("release", vec![arc(1, 1)], vec![arc(0, 1)]),
            ],
            initial_marking: vec![1, 0],
        }
    }

    fn self_loop_fireability_net(initial_tokens: u64) -> PetriNet {
        PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(0, 1)])],
            initial_marking: vec![initial_tokens],
        }
    }

    fn aiger_safety_property(tracker: &PropertyTracker) -> ResolvedPredicate {
        match tracker.quantifier {
            PathQuantifier::AG => tracker.predicate.clone(),
            PathQuantifier::EF => ResolvedPredicate::Not(Box::new(tracker.predicate.clone())),
        }
    }

    fn assert_aiger_unsat_for_tracker(net: &PetriNet, tracker: &PropertyTracker) {
        let bounds = compute_all_place_bounds(net).expect("test net should be bounded");
        let safety_property = aiger_safety_property(tracker);
        let encoding = try_encode_as_aiger(net, &safety_property, &bounds)
            .expect("test property should encode as AIGER");
        let results =
            tla_aiger::check_aiger_sat(&encoding.circuit, Some(std::time::Duration::from_secs(2)));

        assert!(
            matches!(results.first(), Some(tla_aiger::AigerCheckResult::Unsat)),
            "test must exercise AIGER UNSAT, got {results:?}",
        );
    }

    #[test]
    fn test_aiger_trace_to_petri_trace_single_fire() {
        let net = mutex_net();
        let trace_map = replay_trace_map(&net);
        let trace = vec![trace_step(2, &[0]), trace_step(2, &[])];

        let replay =
            aiger_trace_to_petri_trace(&net, &trace_map, &trace).expect("trace should replay");

        assert_eq!(replay.petri_trace, vec![TransitionIdx(0)]);
    }

    #[test]
    fn test_aiger_trace_to_petri_trace_stutter() {
        let net = mutex_net();
        let trace_map = replay_trace_map(&net);
        let trace = vec![trace_step(2, &[]), trace_step(2, &[])];

        let replay =
            aiger_trace_to_petri_trace(&net, &trace_map, &trace).expect("stutter should replay");

        assert!(replay.petri_trace.is_empty());
    }

    #[test]
    fn test_aiger_trace_to_petri_trace_disabled_high_priority_allows_lower_enabled() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1"), place("done")],
            transitions: vec![
                trans("disabled", vec![arc(0, 1)], vec![arc(2, 1)]),
                trans("enabled", vec![arc(1, 1)], vec![arc(2, 1)]),
            ],
            initial_marking: vec![0, 1, 0],
        };
        let trace_map = replay_trace_map(&net);
        let trace = vec![trace_step(2, &[0, 1]), trace_step(2, &[])];

        let replay =
            aiger_trace_to_petri_trace(&net, &trace_map, &trace).expect("trace should replay");

        assert_eq!(replay.petri_trace, vec![TransitionIdx(1)]);
    }

    #[test]
    fn test_aiger_trace_to_petri_trace_multiple_enabled_uses_priority_order() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1"), place("done")],
            transitions: vec![
                trans("first", vec![arc(0, 1)], vec![arc(2, 1)]),
                trans("second", vec![arc(1, 1)], vec![arc(2, 1)]),
            ],
            initial_marking: vec![1, 1, 0],
        };
        let trace_map = replay_trace_map(&net);
        let trace = vec![trace_step(2, &[0, 1]), trace_step(2, &[])];

        let replay =
            aiger_trace_to_petri_trace(&net, &trace_map, &trace).expect("trace should replay");

        assert_eq!(replay.petri_trace, vec![TransitionIdx(0)]);
    }

    #[test]
    fn test_aiger_trace_to_petri_trace_accepts_named_inputs() {
        let net = mutex_net();
        let trace_map = replay_trace_map(&net);
        let mut first = FxHashMap::default();
        first.insert("petri.transition.0".to_string(), 1);
        first.insert("petri.transition.1".to_string(), 0);
        let trace = vec![first, FxHashMap::default()];

        let replay =
            aiger_trace_to_petri_trace(&net, &trace_map, &trace).expect("trace should replay");

        assert_eq!(replay.petri_trace, vec![TransitionIdx(0)]);
    }

    #[test]
    fn test_aiger_trace_latch_cross_check_records_match() {
        let net = mutex_net();
        let trace_map = replay_trace_map(&net);
        let mut first = trace_step(2, &[0]);
        insert_named_latches(&mut first, &trace_map, &[1, 0]);
        let mut second = trace_step(2, &[]);
        insert_named_latches(&mut second, &trace_map, &[0, 1]);
        let trace = vec![first, second];

        let replay =
            aiger_trace_to_petri_trace(&net, &trace_map, &trace).expect("trace should replay");

        assert_eq!(replay.petri_trace, vec![TransitionIdx(0)]);
        assert_eq!(replay.diagnostics.latch_frames_checked, 2);
        assert_eq!(replay.diagnostics.latch_frames_incomplete, 0);
        assert!(replay.diagnostics.latch_mismatches.is_empty());
    }

    #[test]
    fn test_aiger_trace_latch_cross_check_mismatch_is_diagnostic_only() {
        let net = mutex_net();
        let trace_map = replay_trace_map(&net);
        let mut first = trace_step(2, &[0]);
        insert_named_latches(&mut first, &trace_map, &[1, 0]);
        let mut second = trace_step(2, &[]);
        insert_named_latches(&mut second, &trace_map, &[1, 0]);
        let trace = vec![first, second];

        let replay =
            aiger_trace_to_petri_trace(&net, &trace_map, &trace).expect("trace should replay");

        assert_eq!(replay.petri_trace, vec![TransitionIdx(0)]);
        assert_eq!(replay.diagnostics.latch_frames_checked, 2);
        assert!(
            replay
                .diagnostics
                .latch_mismatches
                .iter()
                .any(|mismatch| mismatch.contains("frame 1")),
            "expected frame 1 latch mismatch, got {:?}",
            replay.diagnostics.latch_mismatches,
        );
    }

    #[test]
    fn test_aiger_trace_latch_cross_check_incomplete_is_diagnostic_only() {
        let net = mutex_net();
        let trace_map = replay_trace_map(&net);
        let mut first = trace_step(2, &[0]);
        insert_named_latches(&mut first, &trace_map, &[1, 0]);
        let trace = vec![first, trace_step(2, &[])];

        let replay =
            aiger_trace_to_petri_trace(&net, &trace_map, &trace).expect("trace should replay");

        assert_eq!(replay.petri_trace, vec![TransitionIdx(0)]);
        assert_eq!(replay.diagnostics.latch_frames_checked, 1);
        assert_eq!(replay.diagnostics.latch_frames_incomplete, 1);
        assert!(replay.diagnostics.latch_mismatches.is_empty());
    }

    #[test]
    fn test_aiger_trace_to_petri_trace_rejects_missing_input_key() {
        let net = mutex_net();
        let trace_map = replay_trace_map(&net);
        let mut first = FxHashMap::default();
        first.insert("i0".to_string(), 1);
        let trace = vec![first, trace_step(2, &[])];

        let error = aiger_trace_to_petri_trace(&net, &trace_map, &trace)
            .expect_err("missing i1 should fail closed");

        assert!(error.contains("missing i1"));
    }

    #[test]
    fn test_aiger_seeding_ag_safe_mutex() {
        // Mutex: free + busy conserved = 1. AG(busy <= 1) should be TRUE.
        let net = mutex_net();

        let predicate = ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
            ResolvedIntExpr::Constant(1),
        );
        let mut trackers = vec![make_tracker("AG-mutex", PathQuantifier::AG, predicate)];

        run_aiger_test(&net, &mut trackers);

        assert_eq!(
            trackers[0].verdict,
            Some(true),
            "AG(busy<=1) should be TRUE"
        );
        assert_eq!(
            trackers[0].resolved_by.as_ref().map(|r| r.source),
            Some(ReachabilityResolutionSource::Aiger),
        );
    }

    #[test]
    fn test_aiger_seeding_ef_reachable_validates_or_fails_closed() {
        // Same mutex net. EF(busy >= 1) should be TRUE (acquire fires once).
        let net = mutex_net();

        let predicate = ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(1),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        );
        let mut trackers = vec![make_tracker("EF-busy", PathQuantifier::EF, predicate)];

        run_aiger_test(&net, &mut trackers);

        if trackers[0].verdict.is_some() {
            assert_eq!(
                trackers[0].verdict,
                Some(true),
                "validated EF(busy>=1) witness should be TRUE"
            );
            assert_eq!(
                trackers[0].resolved_by.as_ref().map(|r| r.source),
                Some(ReachabilityResolutionSource::Aiger),
            );
        }
    }

    #[test]
    fn test_aiger_seeding_ef_unreachable() {
        // Conserving net: free + busy = 1. EF(busy >= 2) should be FALSE.
        let net = mutex_net();

        let predicate = ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(2),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
        );
        let mut trackers = vec![make_tracker(
            "EF-unreachable",
            PathQuantifier::EF,
            predicate,
        )];

        run_aiger_test(&net, &mut trackers);

        assert_eq!(
            trackers[0].verdict,
            Some(false),
            "EF(busy>=2) should be FALSE"
        );
    }

    #[test]
    fn test_aiger_seeding_ef_fireability_unsat_stays_unresolved() {
        // t0 is never enabled, so EF(IsFireable(t0)) is false. AIGER can prove
        // UNSAT for that safety query, but fireability proof-side UNSAT is not
        // trusted as an exact reachability verdict.
        let net = self_loop_fireability_net(0);
        let predicate = ResolvedPredicate::IsFireable(vec![TransitionIdx(0)]);
        let mut trackers = vec![make_tracker(
            "EF-fireability-unsat",
            PathQuantifier::EF,
            predicate,
        )];

        assert_aiger_unsat_for_tracker(&net, &trackers[0]);
        run_aiger_test(&net, &mut trackers);

        assert_eq!(trackers[0].verdict, None);
        assert_eq!(trackers[0].resolved_by, None);
    }

    #[test]
    fn test_aiger_seeding_ag_fireability_unsat_stays_unresolved() {
        // The self-loop keeps t0 enabled forever, so AG(IsFireable(t0)) is true.
        // AIGER UNSAT alone still stays unresolved for fireability predicates.
        let net = self_loop_fireability_net(1);
        let predicate = ResolvedPredicate::IsFireable(vec![TransitionIdx(0)]);
        let mut trackers = vec![make_tracker(
            "AG-fireability-unsat",
            PathQuantifier::AG,
            predicate,
        )];

        assert_aiger_unsat_for_tracker(&net, &trackers[0]);
        run_aiger_test(&net, &mut trackers);

        assert_eq!(trackers[0].verdict, None);
        assert_eq!(trackers[0].resolved_by, None);
    }

    #[test]
    fn test_aiger_seeding_nested_fireability_unsat_stays_unresolved() {
        let net = self_loop_fireability_net(1);
        let predicate = ResolvedPredicate::Not(Box::new(ResolvedPredicate::IsFireable(vec![
            TransitionIdx(0),
        ])));
        let mut trackers = vec![make_tracker(
            "EF-nested-fireability-unsat",
            PathQuantifier::EF,
            predicate,
        )];

        assert_aiger_unsat_for_tracker(&net, &trackers[0]);
        run_aiger_test(&net, &mut trackers);

        assert_eq!(trackers[0].verdict, None);
        assert_eq!(trackers[0].resolved_by, None);
    }

    #[test]
    fn test_aiger_seeding_skips_unbounded_net() {
        // Unbounded net: transition produces with no input. AIGER should skip.
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![trans("t0", vec![], vec![arc(0, 1)])],
            initial_marking: vec![0],
        };

        let predicate = ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(1),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(0)]),
        );
        let mut trackers = vec![make_tracker("EF-unbounded", PathQuantifier::EF, predicate)];

        run_aiger_test(&net, &mut trackers);

        assert_eq!(trackers[0].verdict, None, "Should skip unbounded net");
    }

    #[test]
    fn test_aiger_seeding_skips_already_resolved() {
        let net = mutex_net();

        let predicate = ResolvedPredicate::IntLe(
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(1)]),
            ResolvedIntExpr::Constant(1),
        );
        let mut trackers = vec![make_tracker("AG-already", PathQuantifier::AG, predicate)];
        // Pre-resolve
        trackers[0].verdict = Some(true);

        run_aiger_test(&net, &mut trackers);

        // Should remain as-is (no change to resolved_by)
        assert_eq!(trackers[0].resolved_by, None);
    }

    #[test]
    fn test_compute_all_place_bounds_conserving() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![trans("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
            initial_marking: vec![3, 0],
        };

        let bounds = compute_all_place_bounds(&net);
        assert!(bounds.is_some());
        let bounds = bounds.expect("should have bounds");
        assert_eq!(bounds[&PlaceIdx(0)], 3);
        assert_eq!(bounds[&PlaceIdx(1)], 3);
    }

    #[test]
    fn test_compute_all_place_bounds_unbounded() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![trans("t0", vec![], vec![arc(0, 1)])],
            initial_marking: vec![0],
        };

        let bounds = compute_all_place_bounds(&net);
        assert!(bounds.is_none());
    }

    #[test]
    fn test_compute_all_place_bounds_uses_nupn_one_safe_fallback() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0")],
            transitions: vec![trans("t0", vec![], vec![arc(0, 1)])],
            initial_marking: vec![0],
        };
        let nupn = one_place_nupn(&net);

        assert!(compute_all_place_bounds(&net).is_none());
        let bounds = compute_all_place_bounds_with_nupn(&net, Some(&nupn), None)
            .expect("NUPN should supply the one-safe bound");
        assert_eq!(bounds[&PlaceIdx(0)], 1);
    }
}
