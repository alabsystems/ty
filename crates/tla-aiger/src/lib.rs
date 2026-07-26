// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! AIGER format parser, IR, and back ends for bit-level hardware model checking.
//!
//! `tla-aiger` ingests circuits in the [AIGER] And-Inverter Graph format and
//! decides their safety properties. It is the hardware-model-checking front end
//! of the **ty** toolchain (alongside the BTOR2 front end), feeding the same
//! shared SAT/CHC engines that back TLA+ checking.
//!
//! # What it does
//!
//! - **Parse** both the ASCII (`.aag`) and binary (`.aig`) encodings, including
//!   the extended HWMCC header `M I L O A B C J F` (bad, constraints, justice,
//!   fairness) — see [`parser`] and [`types`].
//! - **Lower** a parsed [`AigerCircuit`] either to a CNF [`transys::Transys`]
//!   (for the in-crate SAT engines) or to a CHC problem (for the `ay-chc`
//!   PDR/BMC portfolio) — see [`to_chc`].
//! - **Check** safety with a thread-based [`portfolio`] of engines: bounded
//!   model checking ([`bmc`]), k-induction ([`kind`]), and IC3/PDR ([`ic3`]),
//!   after [`coi`] cone-of-influence reduction and [`preprocess`] simplification.
//!
//! # Entry points
//!
//! - [`parse_file`] / [`parse_aag`] / [`parse_aig`] — read a circuit.
//! - [`check_aiger_sat`] — the preferred one-call check for HWMCC benchmarks
//!   (SAT portfolio, returns one [`AigerCheckResult`] per property).
//! - [`check_aiger`] — CHC-based checking via `ay-chc`.
//! - [`portfolio::portfolio_check`] — direct access to the SAT portfolio with a
//!   custom [`portfolio::PortfolioConfig`].
//!
//! ```no_run
//! use std::time::Duration;
//! use tla_aiger::{parse_file, check_aiger_sat, AigerCheckResult};
//!
//! let circuit = parse_file(std::path::Path::new("counter.aig"))?;
//! for (i, result) in check_aiger_sat(&circuit, Some(Duration::from_secs(60)))
//!     .into_iter()
//!     .enumerate()
//! {
//!     match result {
//!         AigerCheckResult::Unsat => println!("property {i}: SAFE"),
//!         AigerCheckResult::Sat { trace } => {
//!             println!("property {i}: UNSAFE ({} step trace)", trace.len())
//!         }
//!         AigerCheckResult::Unknown { reason } => println!("property {i}: ? ({reason})"),
//!     }
//! }
//! # Ok::<(), tla_aiger::AigerError>(())
//! ```
//!
//! # References and attribution
//!
//! Format: "The AIGER And-Inverter Graph (AIG) Format Version 20071012"
//! by Armin Biere, Johannes Kepler University, 2006-2007.
//!
//! The model-checking engines in this crate (IC3/PDR, BMC, k-induction, the
//! portfolio architecture, and the preprocessing pipeline) primarily follow
//! two papers by Yuheng Su et al.: "Extended CTG Generalization and Dynamic
//! Adjustment of Generalization Strategies in IC3" (arXiv:2501.02480) and
//! "The rIC3 Hardware Model Checker" (arXiv:2502.13605). The rIC3 model
//! checker (github.com/gipsyh/rIC3, GPL-3.0) is gratefully acknowledged as
//! the primary reference implementation for these techniques; all algorithms
//! here are reimplemented independently in Rust under Apache-2.0, with ty's
//! own engineering choices where the papers leave details open.
//!
//! [AIGER]: https://fmv.jku.at/aiger/

// Every public item in this crate is documented; keep it that way.
#![deny(missing_docs)]
// Crate-wide `dead_code` allow: this crate carries a large body of reserved,
// staged-but-not-yet-wired API surface — alternative IC3/PDR strategy helpers,
// portfolio config knobs and tuning constants, and ABI/stats types that are
// constructed only on as-yet-unwired solver paths. These are deliberately kept
// for staged tuning rather than deleted. The hot, partially-wired cases that
// merit a local note already carry their own per-item `#[allow(dead_code)]`
// (e.g. `ic3/vsids.rs`); the blanket allow covers the remaining reserved items
// so they don't have to be individually annotated. Narrowing it to per-item
// annotations would require a full compiler pass over a 50k-line crate to
// enumerate the genuinely-dead set safely, which is intentionally deferred.
#![allow(dead_code)]
// Style preferences:
// - collapsible_if: nested if blocks document the decision tree explicitly
// - manual_clamp: .min(X).max(Y) chains are clearer than .clamp(Y, X), and
//   .clamp panics on misordered bounds.
// - filter_map_identity: `filter_map(|x| f(x))` form mirrors the underlying
//   iterator pipeline more naturally than `map(f)` when `f` returns Option.
#![allow(clippy::collapsible_if)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::filter_map_identity)]
#![allow(clippy::unnecessary_filter_map)]
#![allow(clippy::bind_instead_of_map)]
// Index-based loops document iteration order in IC3/BMC cube/trace machinery.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::explicit_counter_loop)]
#![allow(clippy::while_let_loop)]
#![allow(clippy::only_used_in_recursion)]
#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::doc_lazy_continuation)]
// Internal helper APIs intentionally take `&String` to retain the &Arc-like
// pointer identity for cache keying in the inn_proper/preprocess passes.
#![allow(clippy::ptr_arg)]
// Crate has 5 unused imports that retain ABI types / staged hooks.
#![allow(unused_imports)]
// Inner private types leak through pub enum signatures for portfolio config;
// these are crate-internal config knobs not part of the stable consumer API.
#![allow(private_interfaces)]

pub mod bdd_reach;
pub mod bmc;
pub mod check_result;
pub mod cnf;
pub mod coi;
pub mod error;
pub mod ic3;
pub(crate) mod inn_proper;
pub mod kind;
pub mod parser;
pub mod portfolio;
pub mod preprocess;
pub mod sat_types;
pub mod shared_engine_evidence;
pub mod ternary;
pub mod to_chc;
pub mod transys;
pub mod types;

pub use bmc::witness::extract_original_cex_trace;
pub use bmc::{check_bmc, check_bmc_dynamic, check_kind, check_kind_simple_path, BmcResult};
pub use check_result::CheckResult;
pub use error::AigerError;
pub use parser::{parse_aag, parse_aig, parse_file};
pub use portfolio::{
    aiger_hardware_replay_decision_evidence, aiger_hardware_replay_primitive_status,
    aiger_portfolio_capability_report, balanced_portfolio, cegar_ic3_conservative,
    cegar_ic3_ctp_inf, competition_portfolio, default_preset_pool, full_ic3_portfolio,
    hardware_replay_decision_accepts_replay_primitive, ic3_cegar_const, ic3_cegar_full,
    portfolio_check, portfolio_check_adaptive, portfolio_check_detailed,
    portfolio_check_detailed_with_report, validate_aiger_hardware_replay_decision_evidence,
    validate_aiger_hardware_replay_decision_evidence_row,
    validate_hardware_replay_decision_evidence_row, AdaptivePortfolioConfig, AdaptiveScheduler,
    EngineConfig, HardwareReplayDecisionEvidenceError, HardwareReplayPrimitiveAssignmentStatus,
    HardwareReplayPrimitiveConsumerStatus, HardwareReplayPrimitiveDecisionStatus,
    HardwareReplayPrimitiveRejectionReason, HardwareReplayPrimitiveStatus, PortfolioConfig,
    PortfolioResult, HARDWARE_REPLAY_DECISION_REQUIRED_FIELDS, HARDWARE_REPLAY_DECISION_ROW_KIND,
    HARDWARE_REPLAY_DECISION_SCHEMA, HARDWARE_REPLAY_DECISION_SCHEMA_VERSION,
    HARDWARE_REPLAY_PRIMITIVE_SCHEMA,
};
pub use shared_engine_evidence::{
    aiger_prepared_checker_program, aiger_prepared_program_identity_digest,
    aiger_shared_engine_evidence_rows, AigerSharedEngineEvidence,
};
pub use to_chc::{check_aiger, translate_to_chc, AigerCheckResult, AigerTranslation};
pub use transys::attribute_bad_indices;
pub use types::{AigerAnd, AigerCircuit, AigerJustice, AigerLatch, AigerSymbol, Literal};

use std::time::Duration;

/// Check all safety properties of an AIGER circuit using the SAT-based portfolio
/// (BMC + k-induction). This is the preferred entry point for HWMCC benchmarks.
pub fn check_aiger_sat(circuit: &AigerCircuit, timeout: Option<Duration>) -> Vec<AigerCheckResult> {
    let config = portfolio::PortfolioConfig {
        timeout: timeout.unwrap_or_else(|| Duration::from_secs(3600)),
        ..portfolio::default_portfolio()
    };

    let result = portfolio::portfolio_check(circuit, config);

    // Map the SAT-based CheckResult to the existing AigerCheckResult type
    let n = if !circuit.bad.is_empty() {
        circuit.bad.len()
    } else {
        circuit.outputs.len()
    };

    if n == 0 {
        return vec![];
    }

    match result {
        CheckResult::Safe => (0..n).map(|_| AigerCheckResult::Unsat).collect(),
        CheckResult::Unsafe { trace, depth } => {
            if n == 1 {
                vec![AigerCheckResult::Sat {
                    trace: to_i64_trace(&trace),
                }]
            } else {
                // The portfolio decides the OR of all bad literals, so the
                // violation must be attributed to the specific properties the
                // counterexample actually violates (never blindly to index 0).
                attributed_unsafe_results(circuit, &trace, depth, n)
            }
        }
        CheckResult::Unknown { reason } => (0..n)
            .map(|_| AigerCheckResult::Unknown {
                reason: reason.clone(),
            })
            .collect(),
    }
}

/// Convert a bool-valued witness trace to the i64-valued form carried by
/// [`AigerCheckResult::Sat`].
fn to_i64_trace(
    trace: &[rustc_hash::FxHashMap<String, bool>],
) -> Vec<rustc_hash::FxHashMap<String, i64>> {
    trace
        .iter()
        .map(|step| {
            step.iter()
                .map(|(k, &v)| (k.clone(), i64::from(v)))
                .collect()
        })
        .collect()
}

/// Attribute a multi-property `Unsafe` verdict to the specific bad-state
/// properties its counterexample actually violates, in ORIGINAL circuit terms.
///
/// The SAT portfolio decides the Tseitin OR of all bad literals
/// ([`transys::Transys::get_bad_lit`]), so its `Unsafe` verdict alone does not
/// say WHICH property failed. This helper determines that soundly:
///
/// 1. If `trace` replays on the un-preprocessed system built by
///    [`transys::Transys::from_aiger`] (checked with
///    [`transys::Transys::verify_witness`]), the per-property flags are
///    computed directly from its final step via [`attribute_bad_indices`].
/// 2. Otherwise the trace is keyed to a *preprocessed* system (COI/SCORR/
///    renumbering reshape the latch/input set with no reconstruction map), so
///    a fresh original-frame counterexample is re-derived with bounded model
///    checking via [`extract_original_cex_trace`], bounded by `depth + 1`
///    (property-preserving preprocessing keeps the shortest-counterexample
///    depth). The re-derived trace is itself simulation-verified before use.
///
/// Returns the attributed original-frame trace and one flag per property
/// (`true` = that property's bad literal is TRUE at the trace's final step).
/// Returns `None` when no verified attribution could be made — callers must
/// then report per-property `Unknown` rather than guess (fail-closed).
pub fn attribute_unsafe_to_properties(
    circuit: &AigerCircuit,
    trace: &[rustc_hash::FxHashMap<String, bool>],
    depth: usize,
) -> Option<(Vec<rustc_hash::FxHashMap<String, bool>>, Vec<bool>)> {
    let ts = transys::Transys::from_aiger(circuit);
    let attributed: Vec<rustc_hash::FxHashMap<String, bool>> =
        if !trace.is_empty() && ts.verify_witness(trace).is_ok() {
            trace.to_vec()
        } else {
            extract_original_cex_trace(circuit, depth.saturating_add(1).max(2))?
        };
    let flags = attribute_bad_indices(&ts, &attributed);
    if flags.iter().any(|&violated| violated) {
        Some((attributed, flags))
    } else {
        // A verified counterexample must violate at least one property; an
        // all-false attribution means the trace frame did not line up.
        None
    }
}

/// Map a multi-property `Unsafe` verdict to per-property results: `Sat` (with
/// the attributed trace) for exactly the violated indices, `Unknown` for the
/// rest, and all-`Unknown` when the violation cannot be soundly attributed.
///
/// `trace` is the engine-reported counterexample (possibly keyed to a
/// preprocessed system), `depth` its reported depth, and `n` the number of
/// properties (`circuit.bad.len()`, or `circuit.outputs.len()` when the file
/// has no explicit bad section). See [`attribute_unsafe_to_properties`] for
/// the attribution contract.
pub fn attributed_unsafe_results(
    circuit: &AigerCircuit,
    trace: &[rustc_hash::FxHashMap<String, bool>],
    depth: usize,
    n: usize,
) -> Vec<AigerCheckResult> {
    match attribute_unsafe_to_properties(circuit, trace, depth) {
        Some((attributed, flags)) if flags.len() == n => {
            let i64_trace = to_i64_trace(&attributed);
            flags
                .iter()
                .map(|&violated| {
                    if violated {
                        AigerCheckResult::Sat {
                            trace: i64_trace.clone(),
                        }
                    } else {
                        AigerCheckResult::Unknown {
                            reason: "counterexample violates a different property".into(),
                        }
                    }
                })
                .collect()
        }
        _ => (0..n)
            .map(|_| AigerCheckResult::Unknown {
                reason: "counterexample could not be attributed to a specific property".into(),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression (HIGH soundness): a 2-property circuit with bad0 = constant
    /// FALSE (holds) and bad1 = constant TRUE (violated) used to report
    /// property 0 as Sat unconditionally. The violation must be attributed to
    /// property 1, and property 0 must never be reported violated.
    #[test]
    fn multi_property_unsafe_attributed_to_violated_index() {
        let circuit = parse_aag("aag 0 0 0 0 0 2\n0\n1\n").unwrap();
        let results = check_aiger_sat(&circuit, Some(Duration::from_secs(120)));
        assert_eq!(results.len(), 2);
        assert!(
            !matches!(results[0], AigerCheckResult::Sat { .. }),
            "property 0 (bad = FALSE) must not be reported violated: {:?}",
            results[0]
        );
        assert!(
            matches!(results[1], AigerCheckResult::Sat { .. }),
            "property 1 (bad = TRUE) must be reported violated: {:?}",
            results[1]
        );
    }

    /// A frame-mismatched (here: empty) trace cannot be verified against the
    /// original circuit; the helper must re-derive an original-frame
    /// counterexample via BMC and still attribute correctly.
    #[test]
    fn attribute_unsafe_to_properties_rederives_unverifiable_trace() {
        let circuit = parse_aag("aag 0 0 0 0 0 2\n0\n1\n").unwrap();
        let (trace, flags) =
            attribute_unsafe_to_properties(&circuit, &[], 0).expect("re-derivation must succeed");
        assert!(!trace.is_empty());
        assert_eq!(flags, vec![false, true]);
    }

    /// A verified in-frame trace is attributed directly (no re-derivation).
    #[test]
    fn attribute_unsafe_to_properties_uses_verified_trace() {
        // bad0 = l0 (toggles: violated at step 1), bad1 = l1 (stuck at 0).
        let circuit = parse_aag("aag 2 0 2 0 0 2\n2 3\n4 0\n2\n4\n").unwrap();
        let mut step0 = rustc_hash::FxHashMap::default();
        step0.insert("l0".to_string(), false);
        step0.insert("l1".to_string(), false);
        let mut step1 = rustc_hash::FxHashMap::default();
        step1.insert("l0".to_string(), true);
        step1.insert("l1".to_string(), false);
        let witness = vec![step0, step1];

        let (trace, flags) =
            attribute_unsafe_to_properties(&circuit, &witness, 1).expect("attribution");
        assert_eq!(trace, witness, "verified trace must be used as-is");
        assert_eq!(flags, vec![true, false]);
    }
}
