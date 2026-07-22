// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Solver-level dispatch helpers shared by the native (`translate`) and BMC
//! (`bmc`) equality/compound dispatch paths.
//!
//! The two translators encode compound types very differently (the native path
//! uses `FiniteSetEncoder`/`FunctionEncoder`/`RecordEncoder` over array-of-bool
//! sets and per-variable function maps; the BMC path uses step-indexed arrays,
//! QF_LIA linearization, constant divisors, and EXCEPT-via-store). Those
//! specializations are intentional and stay inline in each module. The helpers
//! here cover only the *bit-for-bit identical* solver term constructions that
//! both paths reach, so the shared logic lives in exactly one place.

use ay_dpll::api::{Solver, Term};
use ay_dpll::SolverError;

/// Encode TLA+ boolean equality `l = r` as `(l /\ r) \/ (~l /\ ~r)`.
///
/// TLA+ has no native boolean-equality connective, so both the native and BMC
/// equality paths expand it identically into this if-and-only-if shape. The
/// BMC set-equality path also reuses this for the per-element membership
/// equivalence `(select S u) = (select T u)`.
pub(crate) fn encode_bool_eq(solver: &mut Solver, l: Term, r: Term) -> Result<Term, SolverError> {
    let a_and_b = solver.try_and(l, r)?;
    let not_l = solver.try_not(l)?;
    let not_r = solver.try_not(r)?;
    let not_a_and_not_b = solver.try_and(not_l, not_r)?;
    solver.try_or(a_and_b, not_a_and_not_b)
}
