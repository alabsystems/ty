// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Symbolic state-equation reachability via ay-chc Constrained Horn Clauses.
//!
//! Tier 3 Item 2: handle Murphy-class blowups (e.g., 4.15E+10 reachable
//! states) where explicit BFS is hopeless but symbolic reasoning is
//! tractable. Encode the Petri-net reachability question as a CHC system
//! and dispatch to [`ay_chc::AdaptivePortfolio`].
//!
//! # Algorithm
//!
//! Following Esparza–Melzer (2000) "Verification of Safety Properties
//! Using Integer Programming: Beyond the State Equation" — the state
//! equation `M = M0 + C·x` gives a necessary condition for reachability.
//! Combined with ay-chc IC3/PDR for completeness on bounded safety
//! properties, this yields a sound symbolic decision procedure:
//!
//! - `Init(M0)` encoded as a CHC fact
//! - For each transition `t`: `M_pre(t) >= arc_weight(t)` guard plus
//!   the linear update `M' = M − pre(t) + post(t)` as a Horn clause
//! - The reachability question for a safety predicate becomes a query:
//!   `Inv(M) ∧ ¬Safety(M) ⇒ false`
//!
//! # Soundness contract
//!
//! - `SAFE`     ⇒ predicate provably holds in all reachable markings.
//! - `UNSAFE`   ⇒ a (validated) counterexample trace exists.
//! - `UNKNOWN`  ⇒ no answer (resource limits, overflow guard tripped,
//!   or solver inconclusive). The dispatcher MUST never invent a verdict.
//!
//! # Overflow safety
//!
//! Per the precedent set by commit `dcd60329` ("invariant overflow"),
//! coefficients from COL nets can have transition arc weights that
//! wrap `u64` once multiplied by cardinality factors. The CHC encoder
//! uses `checked_mul` / `checked_add` for every coefficient computation
//! and returns [`SymbolicVerdict::Unknown`] rather than silently
//! truncating. Wraparound is treated identically to a solver
//! `Unknown` — never overridden with a guess.
//!
//! # Relation to the existing PDR encoder
//!
//! `crates/tla-petri/src/examinations/pdr_encoding.rs` already encodes
//! Petri-net safety as CHC with P-invariant strengthening, stuttering,
//! and a bounded-exact fallback. This module is the **state-equation**
//! flavor: a leaner encoding aimed at the explicit-explosion case
//! (large enabledness fan-out, COL-net cardinality blowups) where
//! falling back to bounded exact BFS is hopeless. It deliberately
//! does *not* duplicate the bounded-exact path — when the symbolic
//! solver returns `Unknown`, the verdict is `Unknown`.

pub(crate) mod chc_dispatch;
pub(crate) mod int_state_equation;
pub(crate) mod reachability_seed;
pub(crate) mod state_equation;

#[cfg(test)]
mod tests;

pub(crate) use chc_dispatch::{symbolic_state_equation_check, SymbolicConfig, SymbolicVerdict};
pub(crate) use reachability_seed::run_symbolic_state_equation_seeding;
