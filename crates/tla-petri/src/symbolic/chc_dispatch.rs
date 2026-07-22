// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Dispatcher from the Esparza–Melzer state-equation encoding to
//! `ay_chc::AdaptivePortfolio`.
//!
//! Builds the CHC system via [`StateEquationEncoder`] and runs the
//! adaptive solver inside ay-chc's verified-result pipeline. The
//! verifier-validated outcome is mapped to a typed
//! [`SymbolicVerdict`]:
//!
//! - `Safe`     → property holds in every reachable marking
//! - `Unsafe`   → counterexample trace (decoded into a witness sequence)
//! - `Unknown`  → solver inconclusive, encoder overflow, or net-too-large
//!
//! The dispatcher never invents a verdict: encoder-side overflow,
//! unsupported predicate shapes, and `VerifiedChcResult::Unknown` all
//! collapse to `Unknown`. This preserves the project-wide invariant
//! that `SAFE`/`UNSAFE` must always be the result of a proof step.

use std::time::Duration;

use ay_chc::{AdaptiveConfig, AdaptivePortfolio, VerifiedChcResult};

use super::state_equation::{
    StateEquationEncoder, StateEquationEncoderError, VarNaming, MAX_SYMBOLIC_PLACES,
};
use crate::petri_net::PetriNet;
use crate::resolved_predicate::ResolvedPredicate;

/// Configuration for the symbolic state-equation dispatcher.
#[derive(Debug, Clone)]
pub(crate) struct SymbolicConfig {
    /// Wall-clock budget for the AdaptivePortfolio. Default: 30 s.
    ///
    /// The portfolio internally splits this across IC3/PDR, BMC,
    /// PDKind, k-induction, and TRL engines, so a single value
    /// suffices for the dispatcher.
    pub time_budget: Duration,
    /// Pass-through to `AdaptiveConfig::with_budget` `verbose` flag.
    pub verbose: bool,
    /// Hard cap on the number of places the encoder will admit. When
    /// the net exceeds this, the dispatcher returns
    /// `Unknown { reason: NetTooLarge }`. Mirrors the constant in
    /// `state_equation.rs` so test code can override locally.
    pub max_places: usize,
}

impl Default for SymbolicConfig {
    fn default() -> Self {
        Self {
            time_budget: Duration::from_secs(30),
            verbose: false,
            max_places: MAX_SYMBOLIC_PLACES,
        }
    }
}

/// One step of a symbolic counterexample witness.
///
/// `Δ` carries the *signed* marking delta inferred from the verified
/// counterexample (`m'[p] − m[p]` for each place that changed). This is
/// enough to replay the trace on the original net for downstream
/// confirmation without requiring the dispatcher to know which
/// transition fired in the underlying solver — the IC3/PDR engine's
/// notion of "transition" is the disjunction across all Petri
/// transitions, not a specific net transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SymbolicWitnessStep {
    pub marking: Vec<i64>,
}

/// Outcome of a symbolic state-equation reachability check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SymbolicVerdict {
    /// Property is provably an invariant of every reachable marking.
    Safe,
    /// A counterexample marking exists. Carries the decoded
    /// witness trace (may be empty when the solver did not surface a
    /// decoded counterexample).
    Unsafe { witness: Vec<SymbolicWitnessStep> },
    /// Solver / encoder could not produce a definite answer. The
    /// caller MUST treat this identically to MCC `CANNOT_COMPUTE` —
    /// never override with a guess.
    Unknown { reason: UnknownReason },
}

/// Diagnostic reason attached to [`SymbolicVerdict::Unknown`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum UnknownReason {
    /// Encoder rejected the net (overflow guard, oversized signature,
    /// or u64-out-of-range token/weight). String body carries the
    /// underlying encoder error for diagnostics.
    EncoderRejected(String),
    /// AdaptivePortfolio finished with a `VerifiedChcResult::Unknown`.
    SolverInconclusive,
    /// Future-non-exhaustive `VerifiedChcResult` variant landed.
    /// Surfaces as Unknown rather than panicking — the project's
    /// soundness floor is "never invent a verdict".
    UnsupportedSolverResult,
}

impl From<StateEquationEncoderError> for UnknownReason {
    fn from(err: StateEquationEncoderError) -> Self {
        Self::EncoderRejected(err.to_string())
    }
}

/// Symbolic state-equation reachability check.
///
/// Encodes `(net, property)` as a CHC system via
/// [`StateEquationEncoder::encode_safety_query`] and runs
/// [`AdaptivePortfolio::solve`] on it. Returns a typed
/// [`SymbolicVerdict`].
///
/// This is the public entry point intended to be called as a
/// **fallback** from explicit BFS examinations (Reachability, OneSafe)
/// when the explicit search is hopeless on Murphy-class blowups. It
/// **never** replaces BFS: callers must run BFS first and only invoke
/// the symbolic check on exhaustion.
#[must_use]
pub(crate) fn symbolic_state_equation_check(
    net: &PetriNet,
    property: &ResolvedPredicate,
    config: &SymbolicConfig,
) -> SymbolicVerdict {
    if net.num_places() > config.max_places {
        return SymbolicVerdict::Unknown {
            reason: UnknownReason::EncoderRejected(format!(
                "net too large: {} places > {} cap",
                net.num_places(),
                config.max_places
            )),
        };
    }

    let encoder = StateEquationEncoder::new(net);
    let problem = match encoder.encode_safety_query(property) {
        Ok(problem) => problem,
        Err(err) => {
            return SymbolicVerdict::Unknown {
                reason: UnknownReason::from(err),
            };
        }
    };

    let adaptive_config = AdaptiveConfig::with_budget(config.time_budget, config.verbose);
    let solver = AdaptivePortfolio::new(problem, adaptive_config);
    let result = solver.solve();

    map_verified_result(result, net.num_places())
}

/// Map a `VerifiedChcResult` into the typed dispatcher verdict.
///
/// Counterexample decoding: ay-chc's `Counterexample` carries a list of
/// steps, each with `(name, i64)` assignment pairs. We project that into
/// `[m_0, m_1, …, m_{P-1}]` per step, using the `se_m_<idx>` naming
/// scheme. Missing variables default to `0` (frame) — this matches the
/// IC3/PDR convention that unmentioned state is unchanged.
fn map_verified_result(result: VerifiedChcResult, num_places: usize) -> SymbolicVerdict {
    match result {
        VerifiedChcResult::Safe(_invariant) => SymbolicVerdict::Safe,
        VerifiedChcResult::Unsafe(verified_cex) => {
            let cex = verified_cex.counterexample();
            let witness = cex
                .steps
                .iter()
                .map(|step| {
                    let mut marking = vec![0_i64; num_places];
                    for p in 0..num_places {
                        if let Some(value) = step.assignments.get(&VarNaming::current_name(p)) {
                            marking[p] = *value;
                        }
                    }
                    SymbolicWitnessStep { marking }
                })
                .collect();
            SymbolicVerdict::Unsafe { witness }
        }
        VerifiedChcResult::Unknown(_marker) => SymbolicVerdict::Unknown {
            reason: UnknownReason::SolverInconclusive,
        },
        _ => SymbolicVerdict::Unknown {
            reason: UnknownReason::UnsupportedSolverResult,
        },
    }
}
