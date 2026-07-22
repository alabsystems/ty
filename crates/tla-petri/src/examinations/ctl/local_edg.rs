// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! CTL adapter atop the unified mu-calculus solver.
//!
//! Historical context: this module formerly contained a 1300-LOC
//! standalone Liu-Smolka extended dependency graph (EDG) CTL solver.
//! That implementation was generalised in two passes (commits
//! `f5930373` introducing the EDG, `92e1fa98` adding EG/AF) and is
//! now a thin translation layer over the unified
//! [`super::super::mu_calculus`] engine.
//!
//! The Liu-Smolka algorithm, the three-valued lattice, the deadline
//! / node-cap / state-cap abort gates, the certain-zero / Dalsgaard
//! tentative-True closure for greatest fixpoints, and the MCC max-
//! path deadlock semantics all live in [`crate::examinations::
//! mu_calculus`] now. This module is a re-export with a thin
//! CTL-flavoured error mapping so the rest of the CTL pipeline does
//! not need to know mu-calculus exists.
//!
//! ## Soundness invariant
//!
//! This adapter is verdict-preserving by construction:
//!
//! - [`ctl_to_mu`](super::super::mu_calculus::ctl_to_mu) is the
//!   standard Emerson-Clarke encoding of CTL in modal mu-calculus,
//!   with the two MCC max-path corrections documented in the
//!   `mu_calculus` module docstring (the `◇true` conjunct on `AF` /
//!   `AU` to exclude deadlocks-without-target from the least fixed
//!   point, and the `¬◇true` disjunct on `EG` to keep
//!   deadlocks-with-predicate inside the greatest fixed point).
//! - The unified solver is structurally identical to the original
//!   CTL EDG, so soundness arguments transfer 1:1; the differential
//!   tests against `tla-mc-core::ctl::CtlEngine` are retained in
//!   both `examinations::ctl::tests::local_edg` (CTL surface) and
//!   `examinations::mu_calculus_tests` (raw mu-calculus surface).

use super::resolve::ResolvedCtl;

use crate::examinations::mu_calculus::{ctl_to_mu, LocalMuSolver, MuAbort};
use crate::explorer::ExplorationConfig;
use crate::petri_net::PetriNet;
use thiserror::Error;

/// Abort modes for the CTL adapter.
///
/// This is the CTL-facing error type. Every variant maps to
/// `Verdict::CannotCompute` at the pipeline boundary; no abort can
/// produce a `True`/`False` verdict.
///
/// The variants mirror [`MuAbort`] but keep the existing CTL
/// pipeline match arms working without churn. In particular,
/// [`Self::UnsupportedOperator`] is retained for legacy
/// compatibility — the unified engine handles every operator the CTL
/// AST can express, so this variant is only produced when the
/// pipeline asks the engine to attempt a formula whose translation
/// would itself be unsound (a future hook; not currently triggered).
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub(super) enum EdgAbort {
    /// EDG node count exceeded the configured cap.
    #[error("local EDG exceeded node cap")]
    NodeCapReached,
    /// Underlying state interner hit its budget.
    #[error("local EDG exceeded state budget")]
    StateLimitReached,
    /// Wall-clock deadline elapsed.
    #[error("local EDG hit the deadline")]
    DeadlineExceeded,
    /// Formula contains an operator the engine cannot handle. With
    /// the unified mu-calculus engine every CTL operator is
    /// supported, so this variant is currently unreachable from the
    /// `ctl_to_mu` path. Retained as the legacy "defer to non-EDG
    /// fallback" signal the pipeline pattern-matches on.
    #[error("local EDG: unsupported CTL operator (defer to fallback checker)")]
    UnsupportedOperator,
}

impl From<MuAbort> for EdgAbort {
    fn from(value: MuAbort) -> Self {
        match value {
            MuAbort::NodeCapReached => EdgAbort::NodeCapReached,
            MuAbort::StateLimitReached => EdgAbort::StateLimitReached,
            MuAbort::DeadlineExceeded => EdgAbort::DeadlineExceeded,
            // #22: a non-representable reachable marking — treat like a state
            // budget decline so the pipeline routes it to CANNOT_COMPUTE.
            MuAbort::TokenOverflow => EdgAbort::StateLimitReached,
            // The remaining MuAbort variants indicate a malformed
            // translation (the CTL→mu encoding produced a formula
            // that the solver rejected). These cannot arise from a
            // well-formed CTL input — `ctl_to_mu` always produces
            // an alternation-free, positive-normal-form mu formula —
            // so they are funnelled to UnsupportedOperator, which
            // the pipeline may route through its secondary fallback.
            MuAbort::NegatedVariable
            | MuAbort::UnboundVariable(_)
            | MuAbort::UnsupportedAlternation => EdgAbort::UnsupportedOperator,
        }
    }
}

/// CTL solver façade. Construct with [`Self::new`], then call
/// [`Self::solve_root`].
///
/// This is a thin wrapper around [`LocalMuSolver`] that owns the
/// translated mu formula (it must outlive the solver call) and
/// converts [`MuAbort`] to [`EdgAbort`].
pub(super) struct LocalEdgSolver<'a> {
    net: &'a PetriNet,
    config: ExplorationConfig,
    node_cap_override: Option<usize>,
}

impl<'a> LocalEdgSolver<'a> {
    pub(super) fn new(net: &'a PetriNet, config: &ExplorationConfig) -> Self {
        Self {
            net,
            config: config.clone(),
            node_cap_override: None,
        }
    }

    #[cfg(test)]
    pub(super) fn with_node_cap(mut self, cap: usize) -> Self {
        self.node_cap_override = Some(cap);
        self
    }

    /// Solve the formula at the initial marking. Translates to mu
    /// internally and delegates to the unified solver.
    pub(super) fn solve_root(&mut self, formula: &ResolvedCtl) -> Result<bool, EdgAbort> {
        let mu_formula = ctl_to_mu(formula);
        let mut solver = LocalMuSolver::new(self.net, &self.config);
        if let Some(cap) = self.node_cap_override {
            solver = solver.with_node_cap(cap);
        }
        solver.solve(&mu_formula).map_err(EdgAbort::from)
    }
}

/// Convenience entry-point used by the pipeline: try to solve
/// `formula` on `net` from the initial marking with the given
/// exploration config.
pub(super) fn solve_local_edg(
    net: &PetriNet,
    formula: &ResolvedCtl,
    config: &ExplorationConfig,
) -> Result<bool, EdgAbort> {
    let mu_formula = ctl_to_mu(formula);
    let mut solver = LocalMuSolver::new(net, config);
    solver.solve(&mu_formula).map_err(EdgAbort::from)
}

/// Node-capped variant used by the pipeline's cheap EDG-first
/// pre-pass: identical engine and verdicts, but aborts (fail-closed,
/// `Err` → fall through to the full-graph path) once `node_cap` EDG
/// nodes are allocated, bounding the pre-pass cost on properties
/// whose certain-zero search cannot decide the root early.
///
/// The cap only ever *lowers* the solver's memory-budgeted default —
/// raising it above the memory budget is never allowed.
pub(super) fn solve_local_edg_capped(
    net: &PetriNet,
    formula: &ResolvedCtl,
    config: &ExplorationConfig,
    node_cap: usize,
) -> Result<bool, EdgAbort> {
    let mu_formula = ctl_to_mu(formula);
    let effective_cap = node_cap.min(LocalMuSolver::memory_budgeted_node_cap(net));
    let mut solver = LocalMuSolver::new(net, config).with_node_cap(effective_cap);
    solver.solve(&mu_formula).map_err(EdgAbort::from)
}
