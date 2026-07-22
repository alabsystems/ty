// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Petri net model checking frontend for `ty`.
//!
//! This crate houses the imported Petri-net MCC engine inside the `ty`
//! workspace. It parses PNML Petri nets, explores their state spaces, exposes
//! MCC examination APIs, and provides a `tla-mc-core` transition-system
//! adapter for generic exploration code.
//!
//! # Techniques
//!
//! - **BFS exploration** with FxHashSet deduplication (configurable `max_states`)
//! - **Structural reductions** — dead-transition, isolated-place,
//!   constant-place, and agglomeration reductions before exploration
//! - **LP state equation** — linear-programming bounds for reachability
//!   and upper-bounds without full state space exploration
//! - **Stubborn sets** — partial-order reduction during BFS
//! - **SCC analysis** — terminal-SCC detection for liveness
//! - **Buchi product** — LTL model checking via automaton product
//! - **CTL checker** — MCC CTL examination support
//! - **Property XML parsing** — CTL, LTL, and reachability formula input
//!
//! # Supported examinations
//!
//! Non-property examinations: ReachabilityDeadlock, OneSafe, StateSpace,
//! QuasiLiveness, StableMarking, and Liveness.
//!
//! Property-XML examinations: UpperBounds, ReachabilityCardinality,
//! ReachabilityFireability, CTLCardinality, CTLFireability,
//! LTLCardinality, and LTLFireability.
//!
//! # Model loading
//!
//! Use [`model::load_model_dir`] as the high-level entry point. It returns
//! a [`model::PreparedModel`] that wraps the parsed net with model name,
//! directory, source net kind, and property alias tables.
//!
//! # Limitations
//!
//! - [`parser::parse_pnml_dir`] remains the strict low-level parser for plain
//!   P/T nets (`ptnet`). Use [`model::load_model_dir`] for the high-level MCC
//!   loading path when supported colored nets (`symmetricnet`) should be
//!   attempted.
//! - Unsupported PNML net kinds or colored-net constructs surface
//!   [`error::PnmlError::UnsupportedNetType`]. The CLI converts those
//!   unsupported-model cases into MCC `CANNOT_COMPUTE` output.
//! - Explicit-state BFS bounded by available memory; LP and structural
//!   techniques extend coverage beyond the BFS horizon.

// Every public item in this crate carries a doc comment; keep it that way.
#![deny(missing_docs)]
// SMT emitter modules build up large `String` buffers via push_str(&format!(..))
// throughout (kinduction, bmc_runner, global_properties_bmc, ltl_lasso_bmc).
// The write!() rewrite is mechanical but invasive across many call sites with
// no perf payoff at SMT-script-generation rates; allow the pattern crate-wide.
#![allow(clippy::format_push_string)]
// Many items (alternative dispatch paths, planned features, certificate variants
// awaiting wiring) are intentionally retained. Centralizing avoids per-impl
// annotation churn.
#![allow(dead_code)]
// Many encode_aiger / lp_state_equation / global_properties_bmc loops index into
// multiple parallel arrays (place_map, transition_map, bits, fire vectors). The
// `for i in 0..n` form is clearer than `(0..n).zip(...).zip(...)` chains and
// matches the SMT/BMC bookkeeping style.
#![allow(clippy::needless_range_loop)]
// Style preferences not worth refactoring:
// - verbose_bit_mask: `x & MASK == 0` is clearer for power-of-two checks than
//   `x.trailing_zeros() >= log2(MASK + 1)`.
// - manual_clamp: `.min(X).max(Y)` chain explicitly tracks intent vs `.clamp(Y, X)`
//   which panics on misordered bounds.
// - format_collect: `iter.map(|x| format!("{x:02x}")).collect()` is readable in
//   short hex-encoding paths; the fold(write!) rewrite is a perf opinion.
// - float_cmp: score/metric equality checks compare exact bit patterns produced
//   by the same code path, not noisy measurements.
// - cmp_owned: one allocation-per-error-path for state-metric value comparison.
#![allow(clippy::verbose_bit_mask)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::format_collect)]
#![allow(clippy::float_cmp)]
#![allow(clippy::cmp_owned)]

pub(crate) mod buchi;
pub(crate) mod circulation_loop;
pub mod cli;

/// Single blessed choke point for process-environment mutation (test/CLI
/// plumbing). Always compiled so in-crate `#[cfg(test)]` tests, the `ty-mcc`
/// binary, and out-of-crate integration tests reach the same choke point. The
/// one `env_mutation` allow lives on `env_guard::raw_env_write`.
#[doc(hidden)]
pub mod env_guard;
pub(crate) mod colored_dead_transitions;
pub(crate) mod colored_reduce;
pub(crate) mod colored_relevance;
pub(crate) mod encode_aiger;
// TODO(#4210): Wire decompose() into examination pipeline; remove allow(dead_code).
#[allow(dead_code)]
pub(crate) mod deadlock_region;
pub(crate) mod decomposition;
pub mod error;
pub mod examination;
pub(crate) mod examinations;
pub mod explorer;
pub(crate) mod formula_simplify;
#[cfg(feature = "gpu")]
pub(crate) mod gpu_state_space;
pub(crate) mod hlpnml;
pub(crate) mod intelligence_bus;
pub(crate) mod invariant;
pub mod liveness_verdict;
pub(crate) mod lp_state_equation;
pub(crate) mod marking;
/// Workspace ay pin validator. Public because the `ty-mcc-ay-pin-validate`
/// binary and `mccctl doctor` both call into this module.
pub mod mcc_ay_pin;
pub(crate) mod mcc_backend_evidence;
/// Generator for the MCC backend-evidence JSONL smoke sidecar. Public
/// because the `ty-mcc-evidence-generate` binary lives in `src/bin/`.
pub mod mcc_backend_evidence_smoke;
/// Shared JSONL iterator for MCC backend-capability evidence sidecars.
/// Used by `ty-mcc-backend-evidence-validate` and
/// `ty-mcc-summarize-evidence` to avoid duplicating the
/// blank/comment-skipping reader with consistent line-number diagnostics.
pub mod mcc_evidence_jsonl;
pub mod mcc_keywords;
/// Numeric-equivalence helper for MCC unit comparison. Public because the
/// `ty-mcc-csv-compare` binary needs it.
pub mod mcc_unit_compare;
pub mod mccctl;
/// Library entry points for the historical `ty-mcc-*` operator helper
/// binaries. Each sub-module exposes a `run()` (parses
/// `std::env::args_os()`) and `run_from(args)` (parses caller-supplied
/// argv). The standalone `src/bin/ty-mcc-*.rs` binaries are 3-line
/// shims that call `run()`; `ty-mccctl <subcommand>` calls
/// `run_from(...)` so we have one compiler-enforced surface.
pub mod mccctl_cmd;
pub(crate) mod memory;
pub mod model;
pub(crate) mod net_class;
pub mod nupn;
pub mod output;
pub mod parser;
pub(crate) mod petri_net;
pub(crate) mod portfolio;
pub(crate) mod property_xml;
pub(crate) mod query_slice;
pub(crate) mod reduction;
pub(crate) mod resolved_predicate;
pub(crate) mod scc;
pub mod simplification_report;
pub(crate) mod structural;
pub(crate) mod stubborn;
pub(crate) mod symbolic;
#[cfg(feature = "dd-backend")]
pub(crate) mod symbolic_colored;
pub mod system;
pub mod timeout;
pub mod tlc_dot;
#[allow(dead_code)]
pub(crate) mod trust_cg_petri_kernel;
/// Trust-VERIFIED StateSpace counting helpers (SMT-backed `Verified` via the
/// Trust verifying compiler `tcargo trust check`; each fn ↔ a machine-checked
/// Lean theorem). Proof-carrying arithmetic seed — see the module header.
pub mod trust_verified_counting;
pub(crate) mod unfold;

// Stable public API: core Petri net model types.
//
// These types are returned by `parser::parse_pnml_dir` and consumed by
// `explorer::ExplorationConfig::auto_sized`, `examination::run_examination`,
// and related functions. Re-exported here so downstream callers use
// `tla_petri::PetriNet` instead of depending on internal module layout.
pub use petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};
pub use system::{CompactMarking, PetriNetSystem, StubbornPorProvider};

/// Process-wide serialization lock for tests that mutate or read process-global
/// environment variables.
///
/// `cargo test` runs all library tests in a single process across many threads.
/// Several tests (and the production code they drive) read or write the same
/// process-global env vars — `AY_PATH`, `HOME`, `PATH`, `BK_TIME_CONFINEMENT`,
/// `BK_MEMORY_CONFINEMENT`, `TY_MCC_BIN`, `TY_MCC_COUPLED_QUOTIENT`, the BMC /
/// k-induction / LTL feature flags, etc. A reader that observes another test's
/// mid-flight mutation flakes intermittently.
///
/// Every env-touching test holds this single lock for its whole duration so two
/// such tests never run concurrently. It must be the ONE canonical lock for the
/// crate: per-module locks only serialize against their own module, which is why
/// the flakiness persisted. RAII guards that restore the prior value on drop
/// (see the `EnvVarGuard` / `with_*_env` helpers) hold this lock so a panicking
/// test cannot leak a mutated var to another test.
#[cfg(test)]
pub(crate) fn env_test_lock() -> std::sync::MutexGuard<'static, ()> {
    use std::sync::{Mutex, OnceLock};

    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
#[path = "wrong_answer_investigation_tests.rs"]
mod wrong_answer_investigation_tests;

#[cfg(test)]
#[path = "upper_bounds_regression_tests.rs"]
mod upper_bounds_regression_tests;

#[cfg(test)]
#[path = "upper_bounds_diagnostic_test.rs"]
mod upper_bounds_diagnostic_test;

#[cfg(test)]
#[path = "one_safe_regression_tests.rs"]
mod one_safe_regression_tests;

#[cfg(test)]
#[path = "siphon_deadlock_crosscheck_tests.rs"]
mod siphon_deadlock_crosscheck_tests;

#[cfg(test)]
#[path = "upper_bounds_exactness_crosscheck_tests.rs"]
mod upper_bounds_exactness_crosscheck_tests;

#[cfg(test)]
#[path = "output_tests.rs"]
mod output_tests;

#[cfg(test)]
#[path = "petri_net_tests.rs"]
mod petri_net_tests;

#[cfg(test)]
#[path = "memory_tests.rs"]
mod memory_tests;

#[cfg(test)]
#[path = "reachability_colored_regression_tests.rs"]
mod reachability_colored_regression_tests;

#[cfg(test)]
#[path = "simplification_report_model_tests.rs"]
mod simplification_report_model_tests;

#[cfg(test)]
#[path = "colored_reduction_tests.rs"]
mod colored_reduction_tests;

#[cfg(all(test, feature = "dd-backend"))]
#[path = "symbolic_colored_tests.rs"]
mod symbolic_colored_tests;
