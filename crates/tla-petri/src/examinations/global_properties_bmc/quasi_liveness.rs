// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::path::Path;
use std::time::{Duration, Instant};

use crate::petri_net::{PetriNet, TransitionIdx};

use super::super::bmc_runner::{emit_bmc_preamble, run_depth_ladder, DepthAction, DepthQuery};
use super::super::reachability_bmc::{
    append_bmc_step_value_query, decode_bmc_model_trace, replay_bmc_trace,
};
use super::super::smt_encoding::{
    find_ay, run_ay_bool_model, SolverBoolModel, SolverOutcome, DEPTH_LADDER, PER_DEPTH_TIMEOUT,
};

/// Minimum budget for a per-transition witness-replay model query. Mirrors the
/// reachability lane's `BMC_SPLIT_RETRY_MIN_BUDGET` floor.
const QL_REPLAY_MIN_BUDGET: Duration = Duration::from_millis(250);

/// Run per-transition BMC for QuasiLiveness.
///
/// Returns a vector of booleans parallel to net.transitions:
/// `true` = transition proven quasi-live (BMC found, and a *replayed* witness
/// confirmed, an enabling state),
/// `false` = unresolved. Does not attempt k-induction (proving
/// "always enabled" is much stronger than "sometimes enabled").
///
/// # Soundness: witness replay
///
/// A transition `t` is quasi-live iff some reachable marking enables `t`. The
/// SMT enabling query is a definite TRUE-direction witness: a SAT means the
/// solver claims a firing sequence from the initial marking that reaches a
/// marking enabling `t`. Like every other symbolic SAT->verdict lane in this
/// crate (reachability_bmc, reachability_aiger), QuasiLiveness must REPLAY that
/// witness on the original net before committing the definite verdict — a
/// spurious solver SAT (e.g. push/pop learned-clause carryover) would otherwise
/// mark a structurally-dead transition quasi-live, flipping the whole
/// examination to a wrong definite TRUE.
///
/// On a raw `Sat`, we re-run the query with model extraction, decode the
/// `stay_*`/`fire_*` model into a concrete firing sequence, replay it on the
/// real net (rejecting any model that fires a disabled transition), and only set
/// `resolved[t] = true` if some replayed marking actually enables `t`. On ANY
/// replay failure the transition is left unresolved (treated as `Unknown`) and
/// handed to the seeded BFS observer, which is exact. This keeps QuasiLiveness
/// answering TRUE on every genuine witness while rejecting spurious SATs.
pub(crate) fn run_quasi_liveness_bmc(net: &PetriNet, deadline: Option<Instant>) -> Vec<bool> {
    let nt = net.num_transitions();
    let mut resolved = vec![false; nt];

    // Transitions with no inputs are always enabled.
    for (i, t) in net.transitions.iter().enumerate() {
        if t.inputs.is_empty() {
            resolved[i] = true;
        }
    }

    let ay_path = match find_ay() {
        Some(p) => p,
        None => return resolved,
    };

    struct QuasiLivenessState<'a> {
        net: &'a PetriNet,
        resolved: &'a mut [bool],
        pending: Vec<usize>,
    }

    let mut state = QuasiLivenessState {
        net,
        resolved: &mut resolved,
        pending: Vec::new(),
    };

    let _ = run_depth_ladder(
        &ay_path,
        DEPTH_LADDER,
        deadline,
        PER_DEPTH_TIMEOUT,
        &mut state,
        |state, depth| {
            state.pending = (0..state.resolved.len())
                .filter(|&transition_idx| !state.resolved[transition_idx])
                .collect();
            if state.pending.is_empty() {
                return None;
            }

            Some(DepthQuery::new(
                encode_quasi_liveness_bmc_script(state.net, &state.pending, depth),
                state.pending.len(),
            ))
        },
        |state, depth, results| match results {
            Some(results) => {
                let mut had_unknown = false;
                for (&transition_idx, outcome) in state.pending.iter().zip(results.iter()) {
                    match outcome {
                        SolverOutcome::Sat => {
                            // Defense-in-depth: never trust a raw solver SAT as a
                            // definite quasi-live verdict. Replay the witness on
                            // the real net (mirrors the reachability lane); only
                            // commit if replay confirms an enabling marking.
                            match validate_quasi_liveness_witness(
                                &ay_path,
                                state.net,
                                transition_idx,
                                depth,
                                deadline,
                            ) {
                                Ok(()) => state.resolved[transition_idx] = true,
                                Err(reason) => {
                                    had_unknown = true;
                                    eprintln!(
                                        "QuasiLiveness BMC depth {depth}: transition \
                                         {transition_idx} SAT model rejected ({reason}); \
                                         leaving unresolved"
                                    );
                                }
                            }
                        }
                        SolverOutcome::Unknown => had_unknown = true,
                        SolverOutcome::Unsat => {}
                    }
                }

                let newly_resolved = state
                    .pending
                    .iter()
                    .filter(|&&transition_idx| state.resolved[transition_idx])
                    .count();
                if newly_resolved > 0 {
                    eprintln!(
                        "QuasiLiveness BMC depth {depth}: {newly_resolved} transitions resolved"
                    );
                }

                if had_unknown {
                    DepthAction::StopDeepening
                } else {
                    DepthAction::Explored
                }
            }
            None => DepthAction::StopDeepening,
        },
    );

    resolved
}

/// Re-run the per-transition enabling query with model extraction, decode the
/// SAT model into a concrete firing sequence, replay it on the original net, and
/// confirm that some replayed marking actually enables the transition.
///
/// `Ok(())` means a genuine witness was reconstructed and validated. `Err(_)`
/// means the SAT was spurious / non-replayable (unparseable model, a fired-but-
/// disabled transition, or no visited marking enables the target) and the caller
/// must NOT commit the definite verdict.
fn validate_quasi_liveness_witness(
    ay_path: &Path,
    net: &PetriNet,
    transition_idx: usize,
    depth: usize,
    deadline: Option<Instant>,
) -> Result<(), String> {
    let timeout = deadline
        .map(|global_deadline| {
            PER_DEPTH_TIMEOUT.min(global_deadline.saturating_duration_since(Instant::now()))
        })
        .unwrap_or(PER_DEPTH_TIMEOUT);
    if timeout < QL_REPLAY_MIN_BUDGET {
        return Err("insufficient model validation budget".to_string());
    }

    let script = encode_quasi_liveness_model_script(net, transition_idx, depth);
    let model = run_ay_bool_model(ay_path, &script, timeout)
        .ok_or_else(|| "solver did not return a parseable SAT model".to_string())?;

    replay_confirms_enablement(net, transition_idx, depth, &model)
}

/// Decode the SAT model into a firing sequence, replay it on the original net,
/// and confirm that some visited marking enables the target transition.
///
/// This is the pure soundness core, kept separate from the solver invocation so
/// it can be unit-tested with constructed models (genuine witnesses must pass;
/// spurious / non-replayable models must be rejected).
fn replay_confirms_enablement(
    net: &PetriNet,
    transition_idx: usize,
    depth: usize,
    model: &SolverBoolModel,
) -> Result<(), String> {
    let trace = decode_bmc_model_trace(net, depth, model)?;
    let markings = replay_bmc_trace(net, &trace)?;

    let target = TransitionIdx(transition_idx as u32);
    if markings
        .iter()
        .any(|marking| net.is_enabled(marking, target))
    {
        Ok(())
    } else {
        Err("replayed trace never enables the target transition".to_string())
    }
}

pub(super) fn encode_quasi_liveness_bmc_script(
    net: &PetriNet,
    pending_transitions: &[usize],
    depth: usize,
) -> String {
    let mut s = String::with_capacity(4096);
    emit_bmc_preamble(&mut s, net, depth);

    // Per-transition check: exists step where transition is enabled
    for &tidx in pending_transitions {
        s.push_str("(push 1)\n");
        push_transition_enabling_assertion(&mut s, net, tidx, depth);
        s.push_str("(check-sat)\n(pop 1)\n");
    }

    s.push_str("(exit)\n");
    s
}

/// Single-transition, model-producing variant of the enabling query.
///
/// Mirrors `reachability_bmc::encode_bmc_model_script`: enables model output,
/// emits the BMC preamble (initial marking + transition relation), asserts the
/// per-transition enabling disjunction, checks sat, then requests the
/// `stay_*`/`fire_*` decision variables so the witness can be decoded and
/// replayed on the real net.
fn encode_quasi_liveness_model_script(net: &PetriNet, tidx: usize, depth: usize) -> String {
    let mut s = String::with_capacity(4096);
    s.push_str("(set-option :produce-models true)\n");
    emit_bmc_preamble(&mut s, net, depth);
    push_transition_enabling_assertion(&mut s, net, tidx, depth);
    s.push_str("(check-sat)\n");
    append_bmc_step_value_query(&mut s, net, depth);
    s.push_str("(exit)\n");
    s
}

/// Assert that transition `tidx` is enabled at some step in `0..=depth`.
fn push_transition_enabling_assertion(s: &mut String, net: &PetriNet, tidx: usize, depth: usize) {
    let transition = &net.transitions[tidx];
    if transition.inputs.is_empty() {
        // Always enabled -- trivially SAT
        s.push_str("(assert true)\n");
        return;
    }

    s.push_str("(assert (or");
    for step in 0..=depth {
        let guards: Vec<String> = transition
            .inputs
            .iter()
            .map(|arc| format!("(>= m_{}_{} {})", step, arc.place.0, arc.weight))
            .collect();
        if guards.len() == 1 {
            s.push_str(&format!(" {}", guards[0]));
        } else {
            s.push_str(&format!(" (and {})", guards.join(" ")));
        }
    }
    s.push_str("))\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examinations::smt_encoding::SolverBoolModel;
    use crate::petri_net::{Arc, PlaceIdx, PlaceInfo, TransitionInfo};

    fn place(id: &str) -> PlaceInfo {
        PlaceInfo {
            id: id.to_string(),
            name: None,
        }
    }

    fn arc(p: u32, w: u64) -> Arc {
        Arc {
            place: PlaceIdx(p),
            weight: w,
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

    /// p0(1) → [t0] → p1 → [t1] → p0. Cyclic; both transitions become enabled
    /// on a genuine firing sequence from the initial marking.
    fn cyclic_net() -> PetriNet {
        PetriNet {
            name: Some("cyclic".to_string()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
                trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
            ],
            initial_marking: vec![1, 0],
        }
    }

    // ── Structural guarantee: the verdict path provably extracts a model ──

    #[test]
    fn model_script_requests_replayable_witness() {
        let net = cyclic_net();
        let script = encode_quasi_liveness_model_script(&net, 1, 2);
        // produce-models so get-value returns the decision variables
        assert!(
            script.contains("(set-option :produce-models true)"),
            "model script must enable model production"
        );
        // the enabling disjunction for the target transition (t1 needs p1>=1)
        assert!(script.contains("(>= m_0_1 1)"));
        assert!(script.contains("(>= m_1_1 1)"));
        assert!(script.contains("(>= m_2_1 1)"));
        // exactly one check-sat (single-property model query)
        assert_eq!(script.matches("(check-sat)").count(), 1);
        // get-value over every stay_*/fire_* decision var so the witness is
        // decodable and replayable
        assert!(script.contains("(get-value ("));
        for step in 0..2 {
            assert!(script.contains(&format!("stay_{step}")));
            assert!(script.contains(&format!("fire_{step}_0")));
            assert!(script.contains(&format!("fire_{step}_1")));
        }
    }

    // ── Genuine witness is accepted (coverage preserved) ──

    #[test]
    fn genuine_witness_replays_and_confirms() {
        let net = cyclic_net();
        // Fire t0 at step 0: p0(1)→p1(1). After that, t1 (needs p1>=1) is
        // enabled. depth=1, so we have step 0 only.
        let model = SolverBoolModel::from_pairs(vec![
            ("stay_0".to_string(), false),
            ("fire_0_0".to_string(), true),
            ("fire_0_1".to_string(), false),
        ]);
        // Target t1 (index 1): genuinely enabled after the replayed firing.
        assert!(
            replay_confirms_enablement(&net, 1, 1, &model).is_ok(),
            "genuine witness enabling t1 must be confirmed"
        );
    }

    #[test]
    fn target_enabled_at_initial_marking_confirms_at_depth_zero() {
        let net = cyclic_net();
        // t0 (index 0) needs p0>=1 and the initial marking is [1,0]: enabled
        // immediately. depth=0: replay yields just the initial marking.
        let model = SolverBoolModel::from_pairs(Vec::<(String, bool)>::new());
        assert!(replay_confirms_enablement(&net, 0, 0, &model).is_ok());
    }

    // ── Spurious SAT shapes are rejected (no wrong definite verdict) ──

    #[test]
    fn spurious_sat_firing_disabled_transition_is_rejected() {
        let net = cyclic_net();
        // Initial marking [1,0]: t1 (needs p1>=1) is DISABLED at step 0.
        // A spurious model claims to fire t1 anyway. replay_bmc_trace must
        // reject the model (fires a disabled transition) → no verdict.
        let model = SolverBoolModel::from_pairs(vec![
            ("stay_0".to_string(), false),
            ("fire_0_0".to_string(), false),
            ("fire_0_1".to_string(), true),
        ]);
        let err = replay_confirms_enablement(&net, 1, 1, &model)
            .expect_err("model firing a disabled transition must be rejected");
        assert!(err.contains("disabled"), "unexpected reason: {err}");
    }

    #[test]
    fn spurious_sat_never_enabling_target_is_rejected() {
        // Net where t1 requires 2 tokens in p1, but no reachable marking within
        // the trace supplies them. A spurious SAT marking t1 quasi-live must be
        // rejected because no replayed marking actually enables it.
        let net = PetriNet {
            name: Some("starved".to_string()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
                // t1 needs 2 tokens in p1 — never reached on this 1-step trace.
                trans("t1", vec![arc(1, 2)], vec![]),
            ],
            initial_marking: vec![1, 0],
        };
        // Replayable, valid trace (fire t0 once: p1 becomes 1, still < 2).
        let model = SolverBoolModel::from_pairs(vec![
            ("stay_0".to_string(), false),
            ("fire_0_0".to_string(), true),
            ("fire_0_1".to_string(), false),
        ]);
        let err = replay_confirms_enablement(&net, 1, 1, &model)
            .expect_err("trace never enabling t1 must be rejected");
        assert!(err.contains("never enables"), "unexpected reason: {err}");
    }

    #[test]
    fn unparseable_model_is_rejected() {
        let net = cyclic_net();
        // Missing the fire_0_* assignments entirely → decode fails → rejected.
        let model = SolverBoolModel::from_pairs(vec![("stay_0".to_string(), false)]);
        assert!(
            replay_confirms_enablement(&net, 1, 1, &model).is_err(),
            "model missing decision variables must be rejected"
        );
    }

    // ── End-to-end: real solver on a genuinely quasi-live net (if available) ──

    #[test]
    fn cyclic_net_all_transitions_resolved_with_replay() {
        // run_quasi_liveness_bmc resolves the solver via find_ay(), which reads
        // the process-global AY_PATH. Hold the crate-wide env lock so a concurrent
        // test that points AY_PATH at a fake solver cannot perturb this run.
        let _env = crate::env_test_lock();
        let net = cyclic_net();
        let resolved = run_quasi_liveness_bmc(&net, None);
        if resolved.iter().any(|&r| r) {
            // ay available: both transitions are genuinely quasi-live, and the
            // replay-validated path must still resolve every one of them.
            assert!(
                resolved.iter().all(|&r| r),
                "all genuinely quasi-live transitions must still resolve after replay"
            );
        } else {
            eprintln!("ay not available, skipping integration test");
        }
    }
}
