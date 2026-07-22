// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! IC3/PDR model checking engine for AIGER circuits.
//!
//! IC3 (Incremental Construction of Inductive Clauses for Indubitable Correctness)
//! is a SAT-based model checking algorithm that proves safety properties by
//! constructing an inductive invariant frame-by-frame.
//!
//! This implementation follows:
//! - Aaron R. Bradley, "SAT-Based Model Checking without Unrolling" (VMCAI 2011)
//! - Niklas Een, Alan Mishchenko, Robert Brayton, "Efficient implementation of
//!   property directed reachability" (FMCAD 2011)
//! - Yuheng Su et al., "Extended CTG Generalization and Dynamic Adjustment of
//!   Generalization Strategies in IC3" (arXiv:2501.02480)
//! - Yuheng Su et al., "The rIC3 Hardware Model Checker" (arXiv:2502.13605)
//!   (see the crate-level attribution note in `lib.rs`)
//!
//! The main loop:
//! 1. Check if Init => !Bad (if init is bad, return UNSAFE at depth 0)
//! 2. Extend frames (add new level)
//! 3. Blocking phase: find bad states reachable at top frame, create obligations
//! 4. Block all obligations (or find counterexample trace)
//! 5. Propagate lemmas forward; if any frame converges, return SAFE
//!
//! # Module structure
//!
//! - `config`: Configuration types (`Ic3Config`, `Ic3Result`, `GeneralizationOrder`, etc.)
//! - `engine`: `Ic3Engine` struct definition, constructor, solver management
//! - `run`: Main IC3 loop (`check`), init checks, state extraction, public entry points
//! - `block`: Blocking phase (`block_all`, `block_one`), CTG parameter computation
//! - `mic`: MIC generalization, CTG, inductiveness checks, domain solver construction
//! - `propagate`: Frame propagation, lemma pushing, infinity frame promotion
//! - `validate`: Independent invariant validation, consecution cross-checks

// --- Core IC3 modules (split from the original monolithic mod.rs) ---
pub(super) mod block;
pub(super) mod config;
pub(super) mod engine;
pub(super) mod mic;
pub(super) mod propagate;
pub(super) mod run;
pub(super) mod validate;

// --- Existing submodules ---
pub(crate) mod cegar;
pub(crate) mod domain;
pub mod frame;
pub mod lift;
pub mod obligation;
pub(crate) mod predprop;
pub(crate) mod vsids;

// --- Public API re-exports ---
pub use config::{GeneralizationOrder, Ic3Config, Ic3Result, RestartStrategy, ValidationStrategy};
pub use engine::Ic3Engine;
pub use run::{check_ic3, check_ic3_with_config};

#[cfg(test)]
pub use run::check_ic3_no_coi;

#[cfg(test)]
mod tests;
