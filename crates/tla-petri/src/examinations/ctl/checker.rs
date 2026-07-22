// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::cell::RefCell;

use super::resolve::ResolvedCtl;

use crate::explorer::FullReachabilityGraph;
use crate::marking::{unpack_marking_config, MarkingConfig};
use crate::petri_net::PetriNet;
use crate::resolved_predicate::{eval_predicate, ResolvedPredicate};
use tla_mc_core::{
    build_predecessor_csr, CsrAdjacency, CtlAtomEvaluator, CtlEngine, IndexedCtlGraph,
};

thread_local! {
    /// Reused scratch for decoding a packed marking during atom evaluation. Thread-
    /// local so the `Send + Sync` evaluator decodes without a shared buffer.
    static UNPACK_SCRATCH: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

/// Evaluates a resolved predicate against a PACKED marking, decoding lazily into a
/// thread-local scratch. Storing packed markings and decoding here — rather than
/// retaining a `Vec<u64>` per state — is the dominant CTL memory win (the codec is
/// lossless, so verdicts are unchanged).
struct PetriAtomEvaluator<'a> {
    net: &'a PetriNet,
    config: &'a MarkingConfig,
}

impl CtlAtomEvaluator<Box<[u8]>, ResolvedPredicate> for PetriAtomEvaluator<'_> {
    #[allow(clippy::borrowed_box)] // signature dictated by `CtlAtomEvaluator<State=Box<[u8]>>`
    fn evaluate(&self, state: &Box<[u8]>, atom: &ResolvedPredicate) -> bool {
        UNPACK_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            unpack_marking_config(state, self.config, &mut scratch);
            eval_predicate(atom, &scratch, self.net)
        })
    }
}

pub(super) struct CtlChecker<'a> {
    full: &'a FullReachabilityGraph,
    net: &'a PetriNet,
    /// Predecessor adjacency in CSR form (audit S4): one flat edge array +
    /// offsets instead of one `Vec<u32>` header + allocation per state.
    /// Same per-state lists as `build_predecessor_adjacency` — a pure
    /// layout swap, so verdicts are unchanged.
    rev_adj: CsrAdjacency,
}

impl<'a> CtlChecker<'a> {
    pub(super) fn new(full: &'a FullReachabilityGraph, net: &'a PetriNet) -> Self {
        let rev_adj = build_predecessor_csr(&full.graph.adj);
        Self { full, net, rev_adj }
    }

    pub(super) fn eval(&self, formula: &ResolvedCtl) -> Vec<bool> {
        self.engine().eval(formula)
    }

    /// Verdict at the initial state (state 0) with top-level early
    /// exit — identical to `self.eval(formula)[0]` by the engine's
    /// `eval_root` contract, but the outermost fixpoint stops the
    /// moment state 0 is decided either way.
    pub(super) fn eval_root(&self, formula: &ResolvedCtl) -> bool {
        self.engine().eval_root(formula)
    }

    fn engine(
        &self,
    ) -> CtlEngine<'_, Box<[u8]>, (u32, u32), ResolvedPredicate, PetriAtomEvaluator<'_>> {
        let graph = IndexedCtlGraph::new_with_csr_predecessors(
            self.full.markings.packed(),
            &self.full.graph.adj,
            &self.rev_adj,
        );
        CtlEngine::new(
            graph,
            PetriAtomEvaluator {
                net: self.net,
                config: self.full.markings.config(),
            },
        )
    }

    #[cfg(test)]
    fn pre_a(&self, sat: &[bool]) -> Vec<bool> {
        self.engine().pre_a(sat)
    }

    #[cfg(test)]
    fn lfp_ef(&self, sat: &[bool]) -> Vec<bool> {
        self.engine().lfp_ef(sat)
    }

    #[cfg(test)]
    fn gfp_eg(&self, sat: &[bool]) -> Vec<bool> {
        self.engine().gfp_eg(sat)
    }

    #[cfg(test)]
    fn lfp_eu(&self, sat_phi: &[bool], sat_psi: &[bool]) -> Vec<bool> {
        self.engine().lfp_eu(sat_phi, sat_psi)
    }
}

#[cfg(test)]
#[path = "checker_tests.rs"]
mod checker_tests;
