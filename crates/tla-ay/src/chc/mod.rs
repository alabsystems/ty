// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! CHC (Constrained Horn Clause) translation for IC3/PDR verification
//!
//! This module translates TLA+ specs to CHC problems for IC3/PDR-based
//! safety verification using the ay-chc crate.
//!
//! # Background
//!
//! IC3/PDR (Property-Directed Reachability) can prove safety properties
//! for infinite-state systems by finding inductive invariants, unlike
//! explicit-state model checking which requires finite enumeration.
//!
//! # CHC Encoding
//!
//! For a TLA+ spec with Init, Next, and Invariant:
//! ```text
//! Clause 1 (Initiation):   Init(vars) => Inv(vars)
//! Clause 2 (Consecution):  Inv(vars) ∧ Next(vars,vars') => Inv(vars')
//! Clause 3 (Query):        Inv(vars) ∧ ¬Safety(vars) => false
//! ```
//!
//! If PDR finds an interpretation for Inv satisfying all clauses, the
//! spec is proven safe for all reachable states.
//!
//! # Example
//!
//! ```no_run
//! use tla_ay::chc::{ChcTranslator, PdrCheckResult};
//! use tla_ay::TlaSort;
//!
//! let mut trans = ChcTranslator::new(&[("count", TlaSort::Int)]).unwrap();
//!
//! // Build Init, Next, Safety expressions and translate them
//! // ...
//!
//! match trans.solve_pdr_default() {
//!     Ok(PdrCheckResult::Safe { invariant }) => println!("Safe: {}", invariant),
//!     Ok(PdrCheckResult::Unsafe { trace }) => println!("Counterexample found"),
//!     Ok(PdrCheckResult::Unknown { reason }) => println!("Unknown: {}", reason),
//!     Err(e) => eprintln!("Error: {}", e),
//! }
//! ```
//!
//! # Current Scope
//!
//! - Scalar Bool/Int/String state variables
//! - Finite-domain function-valued state variables expanded into predicate arguments
//! - Record-valued state variables expanded into per-field predicate arguments
//! - Init → initiation clause
//! - Next → consecution clause (with primed variable handling)
//! - Safety → query clause
//! - `\div`/`%` with a literal positive divisor, or a non-literal
//!   current-state divisor in Next/Safety (discharged by conjoining a
//!   `divisor > 0` side-condition into the safety obligation; a
//!   counterexample to the augmented obligation is only reported Unsafe if a
//!   concrete replay confirms the original property is violated — see
//!   `translation.rs::lower_div_mod`, `builder.rs::finalize_query_clauses`,
//!   and `replay.rs`)
//! - `f[i]` with a literal in-domain index, or a non-literal current-state
//!   index in Next/Safety (discharged the same way, via an
//!   `i ∈ {domain keys}` domain-membership side-condition — see
//!   `translation.rs::translate_func_apply_value`)
//!
//! Copyright 2026 Andrew Yates
//! SPDX-License-Identifier: Apache-2.0

mod builder;
mod replay;
mod result;
mod support;
mod translation;

#[cfg(test)]
mod tests;

pub use ay_chc::{
    ChcProofTranscriptConsumerEvidence, ChcTraceAssignmentEvidence, ChcTraceStepEvidence,
    ChcUnsafeTraceEvidence,
};
pub use result::{
    render_chc_proof_replay_boundary_evidence, PdrCheckResult, PdrProofCheckResult, PdrState,
    AY_CHC_PROOF_REPLAY_BOUNDARY_EXPECTED_FIELDS,
};

use std::collections::HashMap;

use ay_chc::{ChcExpr, ChcProblem, ChcVar, PredicateId};

use crate::TlaSort;

/// Which CHC clause is currently being translated.
///
/// Divisor-positivity handling for `\div`/`%` is context-sensitive (see
/// `ChcTranslator::lower_div_mod` in `translation.rs`): non-literal divisors
/// are accepted with a recorded side-condition only where a
/// "holds in every reachable state" obligation soundly covers every TLC
/// evaluation of the divisor (Next with an unprimed divisor, and Safety).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChcClauseCtx {
    /// Translating the Init predicate (initiation clause).
    Init,
    /// Translating the Next relation (consecution clause).
    Next,
    /// Translating the safety property (query clause).
    Safety,
}

/// Expanded representation of a finite-domain function-valued state variable.
#[derive(Debug, Clone)]
pub struct ChcFuncVarInfo {
    /// Domain elements encoded as string keys.
    pub domain_keys: Vec<String>,
    /// Sort of the function range.
    pub range_sort: TlaSort,
    /// Current-state variables for each function element.
    pub element_vars: HashMap<String, ChcVar>,
    /// Next-state variables for each function element.
    pub element_next_vars: HashMap<String, ChcVar>,
}

/// Expanded representation of a record-valued state variable.
#[derive(Debug, Clone)]
pub struct ChcRecordVarInfo {
    /// Record field names and sorts in canonical name order.
    pub field_sorts: Vec<(String, TlaSort)>,
    /// Current-state variables for each record field.
    pub field_vars: HashMap<String, ChcVar>,
    /// Next-state variables for each record field.
    pub field_next_vars: HashMap<String, ChcVar>,
}

/// CHC translator for IC3/PDR verification
///
/// Translates TLA+ Init/Next/Safety to a CHC problem that can be
/// solved with ay-chc's PDR solver.
pub struct ChcTranslator {
    /// The CHC problem being built
    problem: ChcProblem,
    /// Invariant predicate ID
    inv_pred: PredicateId,
    /// State variable mapping: TLA+ name -> ChcVar (current state)
    vars: HashMap<String, ChcVar>,
    /// State variable mapping: TLA+ name -> ChcVar (next state, primed)
    next_vars: HashMap<String, ChcVar>,
    /// Expanded function-valued state variables: TLA+ name -> per-key vars
    func_vars: HashMap<String, ChcFuncVarInfo>,
    /// Expanded record-valued state variables: TLA+ name -> per-field vars
    record_vars: HashMap<String, ChcRecordVarInfo>,
    /// Flattened invariant predicate arguments for current state
    pred_vars: Vec<ChcVar>,
    /// Flattened invariant predicate arguments for next state
    pred_next_vars: Vec<ChcVar>,
    /// Variable sorts for type checking
    var_sorts: HashMap<String, TlaSort>,
    /// Interned atoms for strings and model values lowered to CHC Ints
    atom_intern: HashMap<String, i64>,
    /// Whether primed variables are allowed in the current translation context
    allow_primed: bool,
    /// Whether scalar lookups should resolve to next-state variables
    use_primed_vars: bool,
    /// Which clause (Init/Next/Safety) is currently being translated
    clause_ctx: ChcClauseCtx,
    /// Well-definedness side-conditions over current-state variables:
    /// divisor-positivity (`divisor > 0`) for `\div`/`%` occurrences whose
    /// divisor is not a positive integer literal, and domain-membership
    /// (`index ∈ {domain keys}`) for `f[i]` occurrences with a non-literal
    /// index. Discharged by conjoining them into the safety obligation at
    /// finalize time (see `finalize_query_clauses`).
    side_conditions: Vec<ChcExpr>,
    /// Translated Init constraints, kept for counterexample replay.
    init_constraints: Vec<ChcExpr>,
    /// Translated Next constraints, kept for counterexample replay.
    next_constraints: Vec<ChcExpr>,
    /// Translated safety constraints. Query-clause materialization is
    /// deferred to `finalize_query_clauses` so side-conditions recorded by
    /// any of Init/Next/Safety translation can augment the obligation.
    safety_constraints: Vec<ChcExpr>,
    /// Whether the deferred query clauses have been materialized.
    finalized: bool,
}
