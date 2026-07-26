// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Counterexample cross-validation for the fused cooperative orchestrator.
//!
//! When a symbolic engine (BMC or PDR) produces a verdict, this module
//! replays the result through the BFS evaluator to confirm agreement
//! before publishing to the shared verdict. This catches soundness bugs
//! in the symbolic translation without crashing the orchestrator.
//!
//! Part of #3836 (F4: Counterexample Cross-Validation).

use std::sync::Arc;
use tla_value::Rp;

use tla_ay::{BmcState, BmcValue};
use tla_core::ast::Module;

use crate::check::CheckResult;
use crate::config::Config;
use crate::eval::EvalCtx;
use crate::state::State;
use crate::value::{FuncValue, Value};

/// Result of cross-validating a symbolic engine's verdict against the BFS evaluator.
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    /// Whether the BFS evaluator agrees with the symbolic engine's verdict.
    pub engine_agrees: bool,
    /// Length of the trace that was cross-validated (0 for PDR safety proofs).
    pub trace_length: usize,
    /// Which symbolic engine produced the verdict being validated.
    pub source_engine: CrossValidationSource,
    /// Human-readable detail about the cross-validation outcome.
    pub detail: String,
    /// Name of the configured invariant the replay confirmed violated at the
    /// final trace state. `Some` only when `engine_agrees` on a counterexample
    /// (never for safety-proof confirmations).
    pub violated_invariant: Option<String>,
    /// The replay-validated counterexample as a structured checker trace,
    /// ready for the standard violation-reporting pipeline (ALIAS transform,
    /// `--trace-format` rendering, the JSON `counterexample` field consumed by
    /// `ty verdict-emit`). `Some` only when `engine_agrees` on a counterexample.
    pub validated_trace: Option<crate::check::Trace>,
}

/// Which symbolic engine produced the verdict being cross-validated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossValidationSource {
    /// BMC (bounded model checking) found a counterexample trace.
    Bmc,
    /// PDR (IC3) proved safety via an inductive invariant.
    Pdr,
    /// k-Induction proved safety via a bounded inductive argument.
    KInduction,
}

impl CrossValidationSource {
    /// Human-readable lane name for reports.
    pub fn lane_name(&self) -> &'static str {
        match self {
            CrossValidationSource::Bmc => "BMC",
            CrossValidationSource::Pdr => "PDR",
            CrossValidationSource::KInduction => "k-Induction",
        }
    }
}

/// Convert a `BmcValue` (symbolic engine representation) to a `Value` (BFS evaluator representation).
///
/// Returns `None` for values that cannot be represented in the BFS evaluator.
pub(crate) fn bmc_value_to_value(bmc_val: &BmcValue) -> Option<Value> {
    match bmc_val {
        BmcValue::Bool(b) => Some(Value::Bool(*b)),
        BmcValue::Int(n) => Some(Value::int(*n)),
        BmcValue::BigInt(n) => Some(Value::big_int(n.clone())),
        BmcValue::String(s) => Some(Value::String(Rp::from(s.as_str()))),
        BmcValue::Set(members) => {
            let converted: Option<Vec<Value>> = members.iter().map(bmc_value_to_value).collect();
            converted.map(Value::set)
        }
        BmcValue::Sequence(elems) => {
            let converted: Option<Vec<Value>> = elems.iter().map(bmc_value_to_value).collect();
            converted.map(Value::seq)
        }
        BmcValue::Record(fields) => {
            let converted: Option<Vec<(String, Value)>> = fields
                .iter()
                .map(|(name, val)| bmc_value_to_value(val).map(|v| (name.clone(), v)))
                .collect();
            converted.map(Value::record)
        }
        BmcValue::Tuple(elems) => {
            let converted: Option<Vec<Value>> = elems.iter().map(bmc_value_to_value).collect();
            converted.map(Value::tuple)
        }
        BmcValue::Function(entries) => {
            let converted: Option<Vec<(Value, Value)>> = entries
                .iter()
                .map(|(k, v)| {
                    let kv = Value::int(*k);
                    bmc_value_to_value(v).map(|vv| (kv, vv))
                })
                .collect();
            converted.map(|pairs| Value::Func(Rp::new(FuncValue::from_sorted_entries(pairs))))
        }
        BmcValue::StringFunction(entries) => {
            let converted: Option<Vec<(Value, Value)>> = entries
                .iter()
                .map(|(key, value)| {
                    let key = Value::String(Rp::from(key.as_str()));
                    bmc_value_to_value(value).map(|value| (key, value))
                })
                .collect();
            converted.map(|pairs| Value::Func(Rp::new(FuncValue::from_sorted_entries(pairs))))
        }
    }
}

/// Cross-validate a BMC counterexample trace by replaying it through the BFS evaluator.
///
/// Thin wrapper over [`cross_validate_symbolic_trace`] with the BMC source tag,
/// kept for the original call sites and tests.
pub fn cross_validate_bmc_trace(
    module: &Module,
    config: &Config,
    trace: &[BmcState],
) -> CrossValidationResult {
    cross_validate_symbolic_trace(module, config, trace, CrossValidationSource::Bmc)
}

/// Convert a full symbolic (BMC-shaped) trace into the checker's structured
/// [`Trace`](crate::check::Trace), one `State` per trace step.
///
/// Returns `None` when any value cannot be represented in the BFS evaluator.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn bmc_states_to_trace(trace: &[BmcState]) -> Option<crate::check::Trace> {
    let states = bmc_states_to_states(trace)?;
    Some(crate::check::Trace::from_states(states))
}

/// Convert every symbolic trace state to an evaluator [`State`].
fn bmc_states_to_states(trace: &[BmcState]) -> Option<Vec<State>> {
    let mut states = Vec::with_capacity(trace.len());
    for state in trace {
        let mut pairs: Vec<(Arc<str>, Value)> = Vec::with_capacity(state.assignments.len());
        for (var_name, bmc_val) in &state.assignments {
            pairs.push((Arc::from(var_name.as_str()), bmc_value_to_value(bmc_val)?));
        }
        // Deterministic variable order (State stores an ordered map anyway,
        // but sort so failure messages are stable).
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        states.push(State::from_pairs(pairs));
    }
    Some(states)
}

/// Cross-validate a symbolic counterexample trace (BMC state shape) by
/// replaying it through the BFS evaluator — the shared oracle check for EVERY
/// symbolic bug-finding lane: BMC violations, k-Induction base-case
/// counterexamples, and PDR unsafe traces (converted via
/// [`pdr_trace_to_bmc_states`]).
///
/// The replay is a FULL soundness gate, not a final-state spot check:
/// 1. EVERY `BmcValue` assignment converts to a `Value` (else inconclusive);
/// 2. the FIRST state must satisfy the configured Init predicate;
/// 3. every consecutive state pair must satisfy the configured Next relation;
/// 4. some configured invariant must evaluate to FALSE at the FINAL state
///    (the violating step: every symbolic lane deepens from depth 0 upward and
///    returns at the first SAT depth).
///
/// Steps 2–3 reject *spurious* SMT models — assignments that violate the
/// invariant but are not executions of the spec at all (e.g. an
/// over-approximate Init lowering admitting `s = {}`). Without them a
/// satisfiable-but-unreachable model was "confirmed" and published as a
/// violation on a genuinely safe spec. Init/Next are always configured for
/// lane-produced traces (the symbolic lanes refuse to solve without them);
/// when a direct caller provides neither, only the final-state invariant
/// check applies (the historical behavior, exercised by unit tests).
///
/// Any conversion failure, replay failure, or evaluation error reports
/// disagreement — the caller must fail closed, never publish.
pub fn cross_validate_symbolic_trace(
    module: &Module,
    config: &Config,
    trace: &[BmcState],
    source: CrossValidationSource,
) -> CrossValidationResult {
    let trace_length = trace.len();
    let lane = source.lane_name();
    let fail = |detail: String| CrossValidationResult {
        engine_agrees: false,
        trace_length,
        source_engine: source,
        detail,
        violated_invariant: None,
        validated_trace: None,
    };

    if trace.is_empty() {
        return fail(format!("{lane} trace is empty — cannot cross-validate"));
    }

    if config.invariants.is_empty() {
        return fail(format!(
            "no invariants configured — cannot verify {lane} violation"
        ));
    }

    // Set up evaluation context with the spec's operators and config constants.
    let mut ctx = EvalCtx::new();
    ctx.load_module(module);
    if let Err(e) = crate::bind_constants_from_config(&mut ctx, config) {
        return fail(format!(
            "cannot bind config constants for {lane} replay: {e} — \
             cross-validation inconclusive"
        ));
    }

    // Convert EVERY trace state (not just the final one) so the whole
    // counterexample can be replayed through Init/Next.
    let Some(states) = bmc_states_to_states(trace) else {
        return fail(format!(
            "cannot convert a {lane} trace value for interpreter replay — \
             cross-validation inconclusive"
        ));
    };

    // SOUNDNESS GATE: replay the full trace as a genuine Init/Next execution.
    let init_next = match (&config.init, &config.next) {
        (Some(init), Some(next)) => {
            let init_name = ctx.resolve_op_name(init).to_string();
            let next_name = ctx.resolve_op_name(next).to_string();
            match (
                ctx.get_op(&init_name).cloned(),
                ctx.get_op(&next_name).cloned(),
            ) {
                (Some(init_def), Some(next_def)) => Some((init_def, next_def)),
                _ => {
                    return fail(format!(
                        "cannot resolve Init/Next operators for {lane} trace replay — \
                         cross-validation inconclusive"
                    ));
                }
            }
        }
        _ => None,
    };
    if let Some((init_def, next_def)) = &init_next {
        let vars: Vec<Arc<str>> = states[0].vars().map(|(name, _)| Arc::clone(name)).collect();
        let mut engine =
            crate::trace_validate::TraceValidationEngine::new(&mut ctx, init_def, next_def, vars);
        match engine.init_holds_on_state(&states[0]) {
            Ok(true) => {}
            Ok(false) => {
                return fail(format!(
                    "{lane} trace state 1 does not satisfy the configured Init predicate — \
                     spurious counterexample rejected (not an execution of the spec)"
                ));
            }
            Err(e) => {
                return fail(format!(
                    "Init replay of {lane} trace state 1 failed to evaluate: {e} — \
                     cross-validation inconclusive"
                ));
            }
        }
        for i in 1..states.len() {
            match engine.next_holds_on_transition(&states[i - 1], &states[i]) {
                Ok(true) => {}
                Ok(false) => {
                    return fail(format!(
                        "{lane} trace transition {} -> {} does not satisfy the configured \
                         Next relation — spurious counterexample rejected (not an \
                         execution of the spec)",
                        i,
                        i + 1
                    ));
                }
                Err(e) => {
                    return fail(format!(
                        "Next replay of {lane} trace transition {} -> {} failed to \
                         evaluate: {e} — cross-validation inconclusive",
                        i,
                        i + 1
                    ));
                }
            }
        }
    }

    // The symbolic lane finds a violation at the last state in the trace.
    // Cross-validate by checking invariants against that final state.
    let final_step = trace[trace.len() - 1].step;
    let final_state = states.last().expect("trace non-empty");
    for (var_name, value) in final_state.vars() {
        ctx.env_mut().insert(Arc::clone(var_name), value.clone());
    }

    // Evaluate each invariant. The symbolic lane claims at least one is violated.
    for inv_name in &config.invariants {
        match ctx.eval_op(inv_name) {
            Ok(Value::Bool(false)) => {
                // Invariant violated — BFS evaluator confirms the symbolic finding.
                return CrossValidationResult {
                    engine_agrees: true,
                    trace_length,
                    source_engine: source,
                    detail: format!(
                        "BFS evaluator confirms invariant '{inv_name}' violated at trace step \
                         {final_step}{}",
                        if init_next.is_some() {
                            " (full Init/Next trace replay OK)"
                        } else {
                            ""
                        }
                    ),
                    violated_invariant: Some(inv_name.clone()),
                    validated_trace: Some(crate::check::Trace::from_states(states.clone())),
                };
            }
            Ok(Value::Bool(true)) => {
                // This invariant holds — continue checking others.
            }
            Ok(other) => {
                return fail(format!(
                    "invariant '{inv_name}' evaluated to non-boolean: {other:?} — \
                     cross-validation inconclusive"
                ));
            }
            Err(e) => {
                return fail(format!(
                    "invariant '{inv_name}' evaluation error: {e} — \
                     cross-validation inconclusive"
                ));
            }
        }
    }

    // All invariants passed — BFS evaluator does NOT confirm the violation.
    fail(format!(
        "BFS evaluator finds all {} invariants hold at trace step {final_step} — \
         {lane} violation not confirmed",
        config.invariants.len(),
    ))
}

/// Panic-safe, fail-closed interpreter confirmation of a symbolic
/// counterexample trace, for use INSIDE symbolic lanes BEFORE they publish a
/// `Violated` verdict to the shared/cooperative race.
///
/// A `Violated` publish truncates the racing BFS lane into a result
/// indistinguishable from a clean `Success`, so it may happen ONLY after the
/// explicit-state evaluator (the permanent oracle) has confirmed the
/// counterexample. A panic during replay (evaluator gap on an exotic value)
/// counts as NOT confirmed — the lane must fail closed (return an
/// inconclusive result, publish nothing) rather than crash its thread or
/// publish an unvalidated claim.
pub fn confirm_symbolic_cex_fail_closed(
    module: &Module,
    config: &Config,
    trace: &[BmcState],
    source: CrossValidationSource,
) -> CrossValidationResult {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cross_validate_symbolic_trace(module, config, trace, source)
    })) {
        Ok(result) => result,
        Err(payload) => {
            let reason = payload
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            CrossValidationResult {
                engine_agrees: false,
                trace_length: trace.len(),
                source_engine: source,
                detail: format!(
                    "cross-validation panicked during interpreter replay: {reason} — \
                     failing closed (counterexample NOT confirmed)"
                ),
                violated_invariant: None,
                validated_trace: None,
            }
        }
    }
}

/// Convert a PDR (CHC) counterexample trace to the BMC state shape so it can
/// be cross-validated by the same interpreter replay as BMC / k-Induction
/// traces. PDR models are integer-valued; every assignment maps to
/// `BmcValue::Int`. Specs whose variables are not integer-modelled simply fail
/// confirmation (fail closed) when the invariant does not evaluate.
pub fn pdr_trace_to_bmc_states(trace: &[tla_ay::chc::PdrState]) -> Vec<BmcState> {
    trace
        .iter()
        .enumerate()
        .map(|(step, state)| BmcState {
            step,
            assignments: state
                .assignments
                .iter()
                .map(|(name, value)| (name.clone(), BmcValue::Int(*value)))
                .collect(),
        })
        .collect()
}

/// Cross-validate a PDR safety proof against the BFS completion status.
///
/// When PDR proves safety (synthesizes an inductive invariant), we check
/// whether BFS also completed successfully as a consistency check. If BFS
/// found a violation while PDR claims safety, that indicates a soundness
/// issue in one of the engines.
///
/// `pdr_invariant` is the synthesized invariant string from PDR (for logging).
pub fn cross_validate_pdr_safety(
    bfs_result: &CheckResult,
    pdr_invariant: &str,
) -> CrossValidationResult {
    match bfs_result {
        CheckResult::Success(stats) => CrossValidationResult {
            engine_agrees: true,
            trace_length: 0,
            source_engine: CrossValidationSource::Pdr,
            detail: format!(
                "BFS completed with {} states, confirming PDR safety proof (invariant: {})",
                stats.states_found,
                truncate_invariant(pdr_invariant),
            ),
            violated_invariant: None,
            validated_trace: None,
        },
        CheckResult::InvariantViolation { invariant, .. } => CrossValidationResult {
            engine_agrees: false,
            trace_length: 0,
            source_engine: CrossValidationSource::Pdr,
            detail: format!(
                "BFS found invariant violation '{}' but PDR claims safety — \
                 possible soundness issue (PDR invariant: {})",
                invariant,
                truncate_invariant(pdr_invariant),
            ),
            violated_invariant: None,
            validated_trace: None,
        },
        CheckResult::PropertyViolation { kind, .. } => CrossValidationResult {
            engine_agrees: false,
            trace_length: 0,
            source_engine: CrossValidationSource::Pdr,
            detail: format!(
                "BFS found property violation ({kind:?}) but PDR claims safety — \
                 possible soundness issue"
            ),
            violated_invariant: None,
            validated_trace: None,
        },
        // BFS hit a limit or other non-definitive result: PDR may still be correct,
        // we just can't confirm it from BFS alone. Report as agreement with caveat.
        _ => CrossValidationResult {
            engine_agrees: true,
            trace_length: 0,
            source_engine: CrossValidationSource::Pdr,
            detail: format!(
                "BFS did not complete (result: {}) — PDR safety proof accepted without \
                 BFS confirmation (invariant: {})",
                bfs_result_summary(bfs_result),
                truncate_invariant(pdr_invariant),
            ),
            violated_invariant: None,
            validated_trace: None,
        },
    }
}

/// Cross-validate a k-induction safety proof against the BFS completion status.
///
/// When k-induction proves safety (inductive step is UNSAT at some depth k),
/// we check whether BFS also completed successfully as a consistency check.
/// If BFS found a violation while k-induction claims safety, that indicates a
/// soundness issue in one of the engines.
///
/// `proved_k` is the induction depth at which the proof succeeded.
pub fn cross_validate_kinduction_safety(
    bfs_result: &CheckResult,
    proved_k: usize,
) -> CrossValidationResult {
    match bfs_result {
        CheckResult::Success(stats) => CrossValidationResult {
            engine_agrees: true,
            trace_length: 0,
            source_engine: CrossValidationSource::KInduction,
            detail: format!(
                "BFS completed with {} states, confirming k-induction safety proof (k={proved_k})",
                stats.states_found,
            ),
            violated_invariant: None,
            validated_trace: None,
        },
        CheckResult::InvariantViolation { invariant, .. } => CrossValidationResult {
            engine_agrees: false,
            trace_length: 0,
            source_engine: CrossValidationSource::KInduction,
            detail: format!(
                "BFS found invariant violation '{invariant}' but k-induction claims safety \
                 at k={proved_k} — possible soundness issue"
            ),
            violated_invariant: None,
            validated_trace: None,
        },
        CheckResult::PropertyViolation { kind, .. } => CrossValidationResult {
            engine_agrees: false,
            trace_length: 0,
            source_engine: CrossValidationSource::KInduction,
            detail: format!(
                "BFS found property violation ({kind:?}) but k-induction claims safety \
                 at k={proved_k} — possible soundness issue"
            ),
            violated_invariant: None,
            validated_trace: None,
        },
        // BFS hit a limit or other non-definitive result: k-induction may still be
        // correct, we just can't confirm it from BFS alone.
        _ => CrossValidationResult {
            engine_agrees: true,
            trace_length: 0,
            source_engine: CrossValidationSource::KInduction,
            detail: format!(
                "BFS did not complete (result: {}) — k-induction safety proof accepted \
                 without BFS confirmation (k={proved_k})",
                bfs_result_summary(bfs_result),
            ),
            violated_invariant: None,
            validated_trace: None,
        },
    }
}

/// Truncate a synthesized invariant string for human-readable output.
fn truncate_invariant(inv: &str) -> String {
    if inv.len() <= 120 {
        inv.to_string()
    } else {
        format!("{}...", &inv[..117])
    }
}

/// One-line summary of a CheckResult variant for logging.
fn bfs_result_summary(result: &CheckResult) -> &'static str {
    match result {
        CheckResult::Success(_) => "success",
        CheckResult::InvariantViolation { .. } => "invariant_violation",
        CheckResult::PropertyViolation { .. } => "property_violation",
        CheckResult::LivenessViolation { .. } => "liveness_violation",
        CheckResult::LimitReached { .. } => "limit_reached",
        _ => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_bmc_value_to_value_bool() {
        assert_eq!(
            bmc_value_to_value(&BmcValue::Bool(true)),
            Some(Value::Bool(true))
        );
        assert_eq!(
            bmc_value_to_value(&BmcValue::Bool(false)),
            Some(Value::Bool(false))
        );
    }

    #[test]
    fn test_bmc_value_to_value_int() {
        assert_eq!(bmc_value_to_value(&BmcValue::Int(42)), Some(Value::int(42)));
        assert_eq!(bmc_value_to_value(&BmcValue::Int(-1)), Some(Value::int(-1)));
    }

    #[test]
    fn test_bmc_value_to_value_set() {
        let bmc_set = BmcValue::Set(vec![BmcValue::Int(1), BmcValue::Int(2)]);
        let result = bmc_value_to_value(&bmc_set);
        assert!(result.is_some());
    }

    #[test]
    fn test_bmc_value_to_value_sequence() {
        let bmc_seq = BmcValue::Sequence(vec![BmcValue::Bool(true), BmcValue::Int(3)]);
        let result = bmc_value_to_value(&bmc_seq);
        assert!(result.is_some());
    }

    #[test]
    fn test_cross_validate_bmc_empty_trace() {
        let module = crate::test_support::parse_module(
            "---- MODULE Empty ----\nVARIABLE x\nInv == TRUE\n====",
        );
        let config = Config {
            invariants: vec!["Inv".to_string()],
            ..Default::default()
        };
        let result = cross_validate_bmc_trace(&module, &config, &[]);
        assert!(!result.engine_agrees);
        assert_eq!(result.trace_length, 0);
        assert_eq!(result.source_engine, CrossValidationSource::Bmc);
    }

    #[test]
    fn test_cross_validate_bmc_no_invariants() {
        let module = crate::test_support::parse_module("---- MODULE NoInv ----\nVARIABLE x\n====");
        let config = Config::default();
        let trace = vec![BmcState {
            step: 0,
            assignments: HashMap::new(),
        }];
        let result = cross_validate_bmc_trace(&module, &config, &trace);
        assert!(!result.engine_agrees);
    }

    #[test]
    fn test_cross_validate_bmc_violation_confirmed() {
        let module = crate::test_support::parse_module(
            "---- MODULE ConfirmViol ----\nVARIABLE x\nInv == x < 2\n====",
        );
        let config = Config {
            invariants: vec!["Inv".to_string()],
            ..Default::default()
        };
        let mut assignments = HashMap::new();
        assignments.insert("x".to_string(), BmcValue::Int(5));
        let trace = vec![BmcState {
            step: 0,
            assignments,
        }];
        let result = cross_validate_bmc_trace(&module, &config, &trace);
        assert!(result.engine_agrees, "detail: {}", result.detail);
        assert_eq!(result.trace_length, 1);
        assert_eq!(result.source_engine, CrossValidationSource::Bmc);
    }

    #[test]
    fn test_cross_validate_bmc_violation_not_confirmed() {
        let module = crate::test_support::parse_module(
            "---- MODULE NoConfirm ----\nVARIABLE x\nInv == x < 10\n====",
        );
        let config = Config {
            invariants: vec!["Inv".to_string()],
            ..Default::default()
        };
        let mut assignments = HashMap::new();
        assignments.insert("x".to_string(), BmcValue::Int(5));
        let trace = vec![BmcState {
            step: 0,
            assignments,
        }];
        let result = cross_validate_bmc_trace(&module, &config, &trace);
        assert!(!result.engine_agrees, "detail: {}", result.detail);
    }

    #[test]
    fn test_cross_validate_pdr_safety_confirmed() {
        let bfs_result = CheckResult::Success(crate::check::CheckStats {
            states_found: 10,
            ..Default::default()
        });
        let result = cross_validate_pdr_safety(&bfs_result, "x \\in {0, 1}");
        assert!(result.engine_agrees);
        assert_eq!(result.source_engine, CrossValidationSource::Pdr);
    }

    #[test]
    fn test_cross_validate_pdr_safety_contradicted() {
        let bfs_result = CheckResult::InvariantViolation {
            invariant: "Inv".to_string(),
            trace: crate::check::Trace::new(),
            stats: crate::check::CheckStats::default(),
        };
        let result = cross_validate_pdr_safety(&bfs_result, "synthesized_inv");
        assert!(!result.engine_agrees);
        assert_eq!(result.source_engine, CrossValidationSource::Pdr);
    }

    #[test]
    fn test_cross_validate_pdr_bfs_limit_reached() {
        let bfs_result = CheckResult::LimitReached {
            limit_type: crate::check::LimitType::States,
            stats: crate::check::CheckStats::default(),
        };
        let result = cross_validate_pdr_safety(&bfs_result, "synthesized_inv");
        // BFS didn't complete — accept PDR proof without confirmation.
        assert!(result.engine_agrees);
    }

    #[test]
    fn test_truncate_invariant_short() {
        assert_eq!(truncate_invariant("short"), "short");
    }

    #[test]
    fn test_truncate_invariant_long() {
        let long = "x".repeat(200);
        let truncated = truncate_invariant(&long);
        assert!(truncated.len() <= 123);
        assert!(truncated.ends_with("..."));
    }
}
