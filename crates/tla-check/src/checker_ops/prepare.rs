// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared preparation helpers for sequential and parallel model checkers.
//!
//! This module is the stable re-export surface for checker setup operations.
//! Both checker paths call these functions to prevent parity drift (the same
//! class of bug that caused #2787).
//!
//! Internal responsibilities are split into child modules:
//! - `operator_setup`: scalar setup helpers (VIEW, expand, inline NEXT, ASSUME)
//! - `module_graph`: import traversal and canonical ops/vars/ASSUME collection
//! - `load`: canonical `EvalCtx` module loading order
//!
//! Part of #810: shared checker setup pipeline.

mod load;
mod module_graph;
mod operator_setup;
mod prepared_program;

pub(crate) use load::load_modules_into_ctx;
pub(crate) use module_graph::collect_ops_vars_and_assumes;
pub(crate) use operator_setup::{
    check_assumes, check_assumes_with_modules, expand_operator_body, lower_inline_next,
    validate_view_operator,
};
// Only exercised by POR unit tests since the per-coverage-action dependency
// extraction landed (audit-2026-07 #11): production POR expands each detected
// action expression via `enumerate::expand_operators_with_primes` instead of
// re-expanding (and re-splitting) the whole Next body.
#[cfg(test)]
pub(crate) use operator_setup::expand_operator_body_with_primes;
pub(crate) use prepared_program::{TlaPreparedProgram, TlaPreparedProgramSource};
