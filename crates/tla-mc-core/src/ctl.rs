// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::collections::VecDeque;
use std::marker::PhantomData;

use crate::traits::{AtomEvaluator, TransitionSystem};

/// CTL formula over caller-defined atom payloads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CtlFormula<Atom> {
    /// Atomic state predicate.
    Atom(Atom),
    /// Boolean negation.
    Not(Box<CtlFormula<Atom>>),
    /// Boolean conjunction.
    And(Vec<CtlFormula<Atom>>),
    /// Boolean disjunction.
    Or(Vec<CtlFormula<Atom>>),
    /// EX(phi): some successor satisfies `phi`.
    EX(Box<CtlFormula<Atom>>),
    /// AX(phi): all successors satisfy `phi`.
    AX(Box<CtlFormula<Atom>>),
    /// EF(phi): some path eventually reaches `phi`.
    EF(Box<CtlFormula<Atom>>),
    /// AF(phi): all paths eventually reach `phi`.
    AF(Box<CtlFormula<Atom>>),
    /// EG(phi): some maximal path stays in `phi`.
    EG(Box<CtlFormula<Atom>>),
    /// AG(phi): all maximal paths stay in `phi`.
    AG(Box<CtlFormula<Atom>>),
    /// E[phi U psi]: some path has `phi` until `psi`.
    EU(Box<CtlFormula<Atom>>, Box<CtlFormula<Atom>>),
    /// A[phi U psi]: all paths have `phi` until `psi`.
    AU(Box<CtlFormula<Atom>>, Box<CtlFormula<Atom>>),
    /// EGF(phi) == E(GF phi): some path visits `phi` infinitely often.
    ///
    /// This is a genuinely NON-CTL fair-cycle operator (it cannot be expressed
    /// with the other operators). It is the carrier for LTL persistence:
    /// `A(FG p) ≡ ¬EGF(¬p)` (recurrence `A(GF p)` is instead handled as the
    /// plain-CTL `AG(AF p)`). Evaluated as the Emerson–Lei greatest fixpoint
    /// `νZ. EFˢ(phi ∧ EXˢ Z)` with the deadlock-stutter successor
    /// `EXˢ(M) = EX(M) ∨ (deadlock ∧ M)`, so a deadlocked `phi`-state is an
    /// infinite `phi`-stutter witness — matching the GPU engine's `CtlOp::EGF`
    /// and the explicit Büchi lane's deadlock self-loop.
    EGF(Box<CtlFormula<Atom>>),
}

/// Accessor for successor state IDs stored in graph edges.
pub trait CtlEdge {
    /// Return the successor state ID referenced by this edge.
    fn successor(&self) -> u32;
}

impl CtlEdge for u32 {
    fn successor(&self) -> u32 {
        *self
    }
}

impl<Label> CtlEdge for (u32, Label) {
    fn successor(&self) -> u32 {
        self.0
    }
}

/// Evaluates CTL atoms against explicit states.
pub trait CtlAtomEvaluator<State, Atom>: Send + Sync {
    /// Evaluate `atom` on `state`.
    fn evaluate(&self, state: &State, atom: &Atom) -> bool;
}

impl<State, Atom, F> CtlAtomEvaluator<State, Atom> for F
where
    F: Fn(&State, &Atom) -> bool + Send + Sync,
{
    fn evaluate(&self, state: &State, atom: &Atom) -> bool {
        self(state, atom)
    }
}

/// Compressed-sparse-row adjacency (audit S4/S6): all edge targets in one
/// flat array, sliced per state via `offsets`. Replaces the per-state
/// `Vec<u32>` representation — removing one 24-byte `Vec` header and one
/// small heap allocation per state — for the long-lived predecessor
/// adjacency of the explicit CTL engine.
///
/// `offsets` is `u32` (not `usize`) deliberately: it halves the offset
/// array and the engine's state-id space is `u32` anyway; construction
/// asserts the cumulative edge count fits.
pub struct CsrAdjacency {
    /// `offsets.len() == state_count + 1`; state `s`'s neighbors are
    /// `edges[offsets[s]..offsets[s+1]]`.
    offsets: Vec<u32>,
    edges: Vec<u32>,
}

impl CsrAdjacency {
    /// Number of states (vertices) this adjacency covers.
    #[must_use]
    pub fn state_count(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The neighbor list of state `s` — identical contents and order to the
    /// corresponding `Vec<u32>` in the nested representation.
    #[must_use]
    pub fn neighbors(&self, s: usize) -> &[u32] {
        &self.edges[self.offsets[s] as usize..self.offsets[s + 1] as usize]
    }
}

/// Predecessor-adjacency view consumed by the CTL engine: either the
/// caller's nested per-state `Vec`s (existing API) or a [`CsrAdjacency`].
/// Same neighbor contents and order either way — a pure layout choice.
enum PredecessorView<'a> {
    Nested(&'a [Vec<u32>]),
    Csr(&'a CsrAdjacency),
}

impl PredecessorView<'_> {
    fn state_count(&self) -> usize {
        match self {
            PredecessorView::Nested(nested) => nested.len(),
            PredecessorView::Csr(csr) => csr.state_count(),
        }
    }

    #[inline]
    fn neighbors(&self, s: usize) -> &[u32] {
        match self {
            PredecessorView::Nested(nested) => &nested[s],
            PredecessorView::Csr(csr) => csr.neighbors(s),
        }
    }
}

/// Indexed explicit graph view consumed by the CTL engine.
pub struct IndexedCtlGraph<'a, State, Edge> {
    states: &'a [State],
    successors: &'a [Vec<Edge>],
    predecessors: PredecessorView<'a>,
}

impl<'a, State, Edge> IndexedCtlGraph<'a, State, Edge> {
    /// Build a graph view over indexed states and adjacency lists.
    ///
    /// The state, successor, and predecessor arrays must all use the same
    /// stable index space.
    #[must_use]
    pub fn new(
        states: &'a [State],
        successors: &'a [Vec<Edge>],
        predecessors: &'a [Vec<u32>],
    ) -> Self {
        Self::with_predecessor_view(states, successors, PredecessorView::Nested(predecessors))
    }

    /// Like [`Self::new`], but with the predecessor adjacency in CSR form
    /// (built by [`build_predecessor_csr`]) — the memory-lean variant for
    /// large graphs.
    #[must_use]
    pub fn new_with_csr_predecessors(
        states: &'a [State],
        successors: &'a [Vec<Edge>],
        predecessors: &'a CsrAdjacency,
    ) -> Self {
        Self::with_predecessor_view(states, successors, PredecessorView::Csr(predecessors))
    }

    fn with_predecessor_view(
        states: &'a [State],
        successors: &'a [Vec<Edge>],
        predecessors: PredecessorView<'a>,
    ) -> Self {
        assert!(
            u32::try_from(states.len()).is_ok(),
            "CTL engine supports at most {} states, got {}",
            u32::MAX,
            states.len()
        );
        assert_eq!(
            successors.len(),
            states.len(),
            "CTL successor adjacency length {} did not match state count {}",
            successors.len(),
            states.len()
        );
        assert_eq!(
            predecessors.state_count(),
            states.len(),
            "CTL predecessor adjacency length {} did not match state count {}",
            predecessors.state_count(),
            states.len()
        );
        Self {
            states,
            successors,
            predecessors,
        }
    }

    /// Return the number of states (vertices) in the indexed CTL graph.
    #[must_use]
    pub fn state_count(&self) -> usize {
        self.states.len()
    }
}

/// Build reverse predecessor adjacency from forward successor adjacency.
#[must_use]
pub fn build_predecessor_adjacency<Edge: CtlEdge>(successors: &[Vec<Edge>]) -> Vec<Vec<u32>> {
    assert!(
        u32::try_from(successors.len()).is_ok(),
        "CTL engine supports at most {} states, got {}",
        u32::MAX,
        successors.len()
    );

    let mut predecessors = vec![Vec::new(); successors.len()];
    for (state, edges) in successors.iter().enumerate() {
        for edge in edges {
            let successor = edge.successor() as usize;
            assert!(
                successor < successors.len(),
                "CTL successor index {} out of bounds for {} states",
                successor,
                successors.len()
            );
            predecessors[successor].push(state as u32);
        }
    }
    predecessors
}

/// Build reverse predecessor adjacency in CSR form via counting sort
/// (audit S4). Per-state neighbor lists are identical — contents AND
/// order — to [`build_predecessor_adjacency`]'s (both fill target buckets
/// in ascending source-state order), without the per-state `Vec` headers,
/// separate allocations, and capacity slack.
#[must_use]
pub fn build_predecessor_csr<Edge: CtlEdge>(successors: &[Vec<Edge>]) -> CsrAdjacency {
    let n = successors.len();
    assert!(
        u32::try_from(n).is_ok(),
        "CTL engine supports at most {} states, got {n}",
        u32::MAX,
    );

    // Pass 1: in-degree per state.
    let mut offsets = vec![0u32; n + 1];
    let mut total: u64 = 0;
    for edges in successors {
        for edge in edges {
            let successor = edge.successor() as usize;
            assert!(
                successor < n,
                "CTL successor index {successor} out of bounds for {n} states",
            );
            offsets[successor + 1] += 1;
            total += 1;
        }
    }
    assert!(
        u32::try_from(total).is_ok(),
        "CTL predecessor CSR supports at most {} edges, got {total}",
        u32::MAX,
    );

    // Prefix sum: offsets[s+1] = end of state s's bucket.
    for s in 0..n {
        offsets[s + 1] += offsets[s];
    }

    // Pass 2: place each edge's source at its target's cursor. `cursor[s]`
    // starts at the bucket base, so within a bucket sources land in the
    // same ascending (source, edge-position) order as the nested builder.
    let mut cursor: Vec<u32> = offsets[..n].to_vec();
    let mut edges_flat = vec![0u32; total as usize];
    for (state, edges) in successors.iter().enumerate() {
        for edge in edges {
            let successor = edge.successor() as usize;
            edges_flat[cursor[successor] as usize] = state as u32;
            cursor[successor] += 1;
        }
    }

    CsrAdjacency {
        offsets,
        edges: edges_flat,
    }
}

/// Reusable CTL fixpoint engine over an indexed explicit graph.
pub struct CtlEngine<'a, State, Edge, Atom, Eval> {
    graph: IndexedCtlGraph<'a, State, Edge>,
    atom_evaluator: Eval,
    atom: PhantomData<fn(&Atom)>,
}

impl<'a, State, Edge, Atom, Eval> CtlEngine<'a, State, Edge, Atom, Eval>
where
    Edge: CtlEdge,
    Eval: CtlAtomEvaluator<State, Atom>,
{
    /// Construct a CTL engine over an indexed graph and atom evaluator.
    #[must_use]
    pub fn new(graph: IndexedCtlGraph<'a, State, Edge>, atom_evaluator: Eval) -> Self {
        Self {
            graph,
            atom_evaluator,
            atom: PhantomData,
        }
    }

    /// Evaluate a CTL formula to a satisfying-state bitset.
    #[must_use]
    pub fn eval(&self, formula: &CtlFormula<Atom>) -> Vec<bool> {
        match formula {
            CtlFormula::Atom(atom) => self.eval_atom(atom),
            CtlFormula::Not(inner) => self.eval(inner).into_iter().map(|value| !value).collect(),
            CtlFormula::And(children) => self.eval_nary_and(children),
            CtlFormula::Or(children) => self.eval_nary_or(children),
            CtlFormula::EX(inner) => {
                let sat = self.eval(inner);
                self.pre_e(&sat)
            }
            CtlFormula::AX(inner) => {
                let sat = self.eval(inner);
                self.pre_a(&sat)
            }
            CtlFormula::EF(inner) => {
                let sat = self.eval(inner);
                self.lfp_ef(&sat)
            }
            CtlFormula::AF(inner) => {
                let sat = self.eval(inner);
                let not_sat: Vec<bool> = sat.iter().map(|&value| !value).collect();
                let eg_not = self.gfp_eg(&not_sat);
                eg_not.into_iter().map(|value| !value).collect()
            }
            CtlFormula::EG(inner) => {
                let sat = self.eval(inner);
                self.gfp_eg(&sat)
            }
            CtlFormula::AG(inner) => {
                let sat = self.eval(inner);
                let not_sat: Vec<bool> = sat.iter().map(|&value| !value).collect();
                let ef_not = self.lfp_ef(&not_sat);
                ef_not.into_iter().map(|value| !value).collect()
            }
            CtlFormula::EU(phi, psi) => {
                let sat_phi = self.eval(phi);
                let sat_psi = self.eval(psi);
                self.lfp_eu(&sat_phi, &sat_psi)
            }
            CtlFormula::AU(phi, psi) => {
                let sat_phi = self.eval(phi);
                let sat_psi = self.eval(psi);
                let not_phi: Vec<bool> = sat_phi.iter().map(|&value| !value).collect();
                let not_psi: Vec<bool> = sat_psi.iter().map(|&value| !value).collect();
                let not_phi_and_not_psi: Vec<bool> = (0..self.state_count())
                    .map(|index| not_phi[index] && not_psi[index])
                    .collect();
                let eu = self.lfp_eu(&not_psi, &not_phi_and_not_psi);
                let eg = self.gfp_eg(&not_psi);
                (0..self.state_count())
                    .map(|index| !(eu[index] || eg[index]))
                    .collect()
            }
            CtlFormula::EGF(inner) => {
                let sat = self.eval(inner);
                self.gfp_egf(&sat)
            }
        }
    }

    /// Evaluate a CTL formula at state 0 (the initial state) only,
    /// with *top-level early exit*.
    ///
    /// Returns exactly `self.eval(formula)[0]` — verdict-identical by
    /// construction — but the outermost fixpoint stops the instant
    /// state 0 is decided:
    ///
    /// - least fixpoints (`EF`/`EU` and the duals' inner `lfp`) only
    ///   ever *add* states, so the moment state 0 enters the set its
    ///   membership in the final fixpoint is certain (early `true`);
    ///   if the BFS drains without adding it, it is certainly out.
    /// - greatest fixpoints (`EG` and the duals' inner `gfp`) only
    ///   ever *remove* states, so the moment state 0 is eliminated
    ///   its absence from the final fixpoint is certain (early
    ///   `false`); if elimination drains with state 0 surviving, it
    ///   is certainly in.
    ///
    /// Early exit is applied ONLY at the top level (and recursively
    /// through top-level `Not`/`And`/`Or`, which need just the
    /// state-0 bit of their children): inner temporal operators feed
    /// full bitsets to their parents and are evaluated exhaustively
    /// via [`Self::eval`].
    ///
    /// # Panics
    ///
    /// Panics if the graph is empty (there is no state 0).
    #[must_use]
    pub fn eval_root(&self, formula: &CtlFormula<Atom>) -> bool {
        assert!(
            self.state_count() > 0,
            "eval_root requires a non-empty graph"
        );
        match formula {
            CtlFormula::Atom(atom) => self.atom_evaluator.evaluate(&self.graph.states[0], atom),
            CtlFormula::Not(inner) => !self.eval_root(inner),
            // Boolean short-circuit over the state-0 bit: identical
            // to the bitwise and/or of full child bitsets at index 0.
            CtlFormula::And(children) => children.iter().all(|child| self.eval_root(child)),
            CtlFormula::Or(children) => children.iter().any(|child| self.eval_root(child)),
            CtlFormula::EX(inner) => {
                let sat = self.eval(inner);
                self.graph.successors[0]
                    .iter()
                    .any(|edge| sat[edge.successor() as usize])
            }
            CtlFormula::AX(inner) => {
                let sat = self.eval(inner);
                self.graph.successors[0]
                    .iter()
                    .all(|edge| sat[edge.successor() as usize])
            }
            CtlFormula::EF(inner) => {
                let sat = self.eval(inner);
                self.lfp_ef_reaches_root(&sat)
            }
            CtlFormula::AF(inner) => {
                // AF(phi) == NOT EG(NOT phi); early exit when state 0
                // is eliminated from the EG gfp (=> AF true).
                let sat = self.eval(inner);
                let not_sat: Vec<bool> = sat.iter().map(|&value| !value).collect();
                !self.gfp_eg_holds_at_root(&not_sat)
            }
            CtlFormula::EG(inner) => {
                let sat = self.eval(inner);
                self.gfp_eg_holds_at_root(&sat)
            }
            CtlFormula::AG(inner) => {
                // AG(phi) == NOT EF(NOT phi); early exit when the
                // backward reachability touches state 0 (=> AG false).
                let sat = self.eval(inner);
                let not_sat: Vec<bool> = sat.iter().map(|&value| !value).collect();
                !self.lfp_ef_reaches_root(&not_sat)
            }
            CtlFormula::EU(phi, psi) => {
                let sat_phi = self.eval(phi);
                let sat_psi = self.eval(psi);
                self.lfp_eu_reaches_root(&sat_phi, &sat_psi)
            }
            CtlFormula::AU(phi, psi) => {
                // Same algebraic identity as `eval`:
                // A[phi U psi] == NOT (E[!psi U (!phi & !psi)] OR EG(!psi)),
                // with each disjunct early-exiting at state 0 and the
                // second skipped entirely when the first already
                // decides the verdict.
                let sat_phi = self.eval(phi);
                let sat_psi = self.eval(psi);
                let not_phi_and_not_psi: Vec<bool> = (0..self.state_count())
                    .map(|index| !sat_phi[index] && !sat_psi[index])
                    .collect();
                let not_psi: Vec<bool> = sat_psi.iter().map(|&value| !value).collect();
                if self.lfp_eu_reaches_root(&not_psi, &not_phi_and_not_psi) {
                    return false;
                }
                !self.gfp_eg_holds_at_root(&not_psi)
            }
            CtlFormula::EGF(inner) => {
                // No early-exit specialization: EGF is a nested fixpoint whose
                // state-0 membership is only settled at outer-gfp convergence.
                // Compute the full satisfying set and project — verdict-identical
                // to `self.eval(formula)[0]`.
                let sat = self.eval(inner);
                self.gfp_egf(&sat)[0]
            }
        }
    }

    /// State-0 projection of [`Self::lfp_ef`] with early exit: the
    /// lfp only adds states, so membership of state 0 is final the
    /// moment it is inserted.
    fn lfp_ef_reaches_root(&self, sat: &[bool]) -> bool {
        if sat[0] {
            return true;
        }
        let mut result = sat.to_vec();
        let mut queue: VecDeque<u32> = VecDeque::new();
        for (state, &is_sat) in result.iter().enumerate() {
            if is_sat {
                queue.push_back(state as u32);
            }
        }
        while let Some(state) = queue.pop_front() {
            for &predecessor in self.graph.predecessors.neighbors(state as usize) {
                if !result[predecessor as usize] {
                    if predecessor == 0 {
                        return true;
                    }
                    result[predecessor as usize] = true;
                    queue.push_back(predecessor);
                }
            }
        }
        false
    }

    /// State-0 projection of [`Self::lfp_eu`] with early exit.
    fn lfp_eu_reaches_root(&self, sat_phi: &[bool], sat_psi: &[bool]) -> bool {
        if sat_psi[0] {
            return true;
        }
        let mut result = sat_psi.to_vec();
        let mut queue: VecDeque<u32> = VecDeque::new();
        for (state, &is_sat) in result.iter().enumerate() {
            if is_sat {
                queue.push_back(state as u32);
            }
        }
        while let Some(state) = queue.pop_front() {
            for &predecessor in self.graph.predecessors.neighbors(state as usize) {
                let predecessor_index = predecessor as usize;
                if !result[predecessor_index] && sat_phi[predecessor_index] {
                    if predecessor == 0 {
                        return true;
                    }
                    result[predecessor_index] = true;
                    queue.push_back(predecessor);
                }
            }
        }
        false
    }

    /// State-0 projection of [`Self::gfp_eg`] with early exit: the
    /// gfp only removes states, so the elimination of state 0 is
    /// final the moment it happens.
    fn gfp_eg_holds_at_root(&self, sat: &[bool]) -> bool {
        if !sat[0] {
            return false;
        }
        let mut current = sat.to_vec();
        let mut succ_in_set: Vec<u32> = vec![0; self.state_count()];
        let mut queue: VecDeque<u32> = VecDeque::new();

        for state in 0..self.state_count() {
            if !current[state] {
                continue;
            }
            if self.graph.successors[state].is_empty() {
                succ_in_set[state] = u32::MAX;
                continue;
            }
            let count = self.graph.successors[state]
                .iter()
                .filter(|edge| current[edge.successor() as usize])
                .count() as u32;
            succ_in_set[state] = count;
            if count == 0 {
                queue.push_back(state as u32);
            }
        }

        while let Some(state) = queue.pop_front() {
            let state_idx = state as usize;
            if !current[state_idx] {
                continue;
            }
            if state == 0 {
                return false;
            }
            current[state_idx] = false;
            for &predecessor in self.graph.predecessors.neighbors(state_idx) {
                let pred_idx = predecessor as usize;
                if !current[pred_idx] || succ_in_set[pred_idx] == u32::MAX {
                    continue;
                }
                succ_in_set[pred_idx] = succ_in_set[pred_idx].saturating_sub(1);
                if succ_in_set[pred_idx] == 0 {
                    queue.push_back(predecessor);
                }
            }
        }

        current[0]
    }

    /// EX: states with some successor satisfying `sat`.
    ///
    /// Deadlock states have no successors, so `EX` is false there.
    #[must_use]
    pub fn pre_e(&self, sat: &[bool]) -> Vec<bool> {
        self.assert_sat_len(sat);

        let mut result = vec![false; self.state_count()];
        for (res, adj) in result.iter_mut().zip(self.graph.successors.iter()) {
            if !adj.is_empty() {
                for edge in adj {
                    if sat[edge.successor() as usize] {
                        *res = true;
                        break;
                    }
                }
            }
        }
        result
    }

    /// AX: states where all successors satisfy `sat`.
    ///
    /// Deadlock states have no successors, so `AX` is vacuously true there.
    #[must_use]
    pub fn pre_a(&self, sat: &[bool]) -> Vec<bool> {
        self.assert_sat_len(sat);

        let mut result = vec![false; self.state_count()];
        for (res, adj) in result.iter_mut().zip(self.graph.successors.iter()) {
            if adj.is_empty() {
                *res = true;
            } else {
                *res = adj.iter().all(|edge| sat[edge.successor() as usize]);
            }
        }
        result
    }

    /// EF: backward BFS from `sat` states (least fixpoint μZ. sat ∨ EX(Z)).
    #[must_use]
    pub fn lfp_ef(&self, sat: &[bool]) -> Vec<bool> {
        self.assert_sat_len(sat);

        let mut result = sat.to_vec();
        let mut queue: VecDeque<u32> = VecDeque::new();
        for (state, &is_sat) in result.iter().enumerate() {
            if is_sat {
                queue.push_back(state as u32);
            }
        }
        while let Some(state) = queue.pop_front() {
            for &predecessor in self.graph.predecessors.neighbors(state as usize) {
                if !result[predecessor as usize] {
                    result[predecessor as usize] = true;
                    queue.push_back(predecessor);
                }
            }
        }
        result
    }

    /// EG: greatest fixpoint νZ. sat ∧ EX(Z).
    ///
    /// Deadlock states use maximal-path semantics: they remain in the result
    /// iff `sat` already holds there.
    #[must_use]
    pub fn gfp_eg(&self, sat: &[bool]) -> Vec<bool> {
        self.assert_sat_len(sat);

        let mut current = sat.to_vec();
        let mut succ_in_set: Vec<u32> = vec![0; self.state_count()];
        let mut queue: VecDeque<u32> = VecDeque::new();

        for state in 0..self.state_count() {
            if !current[state] {
                continue;
            }
            if self.graph.successors[state].is_empty() {
                succ_in_set[state] = u32::MAX;
                continue;
            }
            let count = self.graph.successors[state]
                .iter()
                .filter(|edge| current[edge.successor() as usize])
                .count() as u32;
            succ_in_set[state] = count;
            if count == 0 {
                queue.push_back(state as u32);
            }
        }

        while let Some(state) = queue.pop_front() {
            let state_idx = state as usize;
            if !current[state_idx] {
                continue;
            }
            current[state_idx] = false;
            for &predecessor in self.graph.predecessors.neighbors(state_idx) {
                let pred_idx = predecessor as usize;
                if !current[pred_idx] || succ_in_set[pred_idx] == u32::MAX {
                    continue;
                }
                succ_in_set[pred_idx] = succ_in_set[pred_idx].saturating_sub(1);
                if succ_in_set[pred_idx] == 0 {
                    queue.push_back(predecessor);
                }
            }
        }

        current
    }

    /// EXˢ: the deadlock-stutter existential successor,
    /// `EXˢ(sat) = EX(sat) ∨ (deadlock ∧ sat)`.
    ///
    /// Identical to [`Self::pre_e`] on states that have successors; at a
    /// deadlock state (no successors) the state's own `sat` bit counts — the
    /// infinite self-stutter. This is the successor relation the fair-cycle
    /// [`Self::gfp_egf`] iterates so a deadlocked witness is not silently
    /// dropped (matching the GPU `CtlOp::EGF` engine's `deadlock ∧ M` term and
    /// the Büchi lane's deadlock self-loop).
    fn pre_e_stutter(&self, sat: &[bool]) -> Vec<bool> {
        let mut result = self.pre_e(sat);
        for (state, res) in result.iter_mut().enumerate() {
            if self.graph.successors[state].is_empty() && sat[state] {
                *res = true;
            }
        }
        result
    }

    /// EGF: `E(GF phi)` — some path visits `phi` infinitely often.
    ///
    /// The Emerson–Lei fair-cycle greatest fixpoint `νZ. EFˢ(phi ∧ EXˢ Z)`
    /// with the deadlock-stutter successor EXˢ ([`Self::pre_e_stutter`]). The
    /// inner `EFˢ(base) = μY. base ∨ EXˢ Y` coincides with [`Self::lfp_ef`]
    /// (under either successor a deadlock contributes only its own `base` bit),
    /// so the inner least fixpoint is exactly backward reachability from `base`.
    /// `Z` starts all-true and shrinks monotonically to the states that begin a
    /// `phi`-fair path; the sequence is decreasing and finite, so it converges.
    ///
    /// Verdict-identical to the GPU `CtlOp::EGF` engine (same stutter pin): a
    /// deadlocked `phi`-state is a fair witness on both, `¬phi` never-recurring
    /// tails are excluded on both.
    #[must_use]
    pub fn gfp_egf(&self, sat: &[bool]) -> Vec<bool> {
        self.assert_sat_len(sat);

        let n = self.state_count();
        let mut current = vec![true; n];
        loop {
            let ex_z = self.pre_e_stutter(&current);
            let base: Vec<bool> = (0..n).map(|index| sat[index] && ex_z[index]).collect();
            let next = self.lfp_ef(&base);
            if next == current {
                return current;
            }
            current = next;
        }
    }

    /// E[phi U psi]: backward BFS from `psi` states through `phi` states.
    #[must_use]
    pub fn lfp_eu(&self, sat_phi: &[bool], sat_psi: &[bool]) -> Vec<bool> {
        self.assert_sat_len(sat_phi);
        self.assert_sat_len(sat_psi);

        let mut result = sat_psi.to_vec();
        let mut queue: VecDeque<u32> = VecDeque::new();
        for (state, &is_sat) in result.iter().enumerate() {
            if is_sat {
                queue.push_back(state as u32);
            }
        }
        while let Some(state) = queue.pop_front() {
            for &predecessor in self.graph.predecessors.neighbors(state as usize) {
                let predecessor_index = predecessor as usize;
                if !result[predecessor_index] && sat_phi[predecessor_index] {
                    result[predecessor_index] = true;
                    queue.push_back(predecessor);
                }
            }
        }
        result
    }

    fn state_count(&self) -> usize {
        self.graph.state_count()
    }

    fn assert_sat_len(&self, sat: &[bool]) {
        assert_eq!(
            sat.len(),
            self.state_count(),
            "CTL bitset length {} did not match graph state count {}",
            sat.len(),
            self.state_count()
        );
    }

    fn eval_atom(&self, atom: &Atom) -> Vec<bool> {
        self.graph
            .states
            .iter()
            .map(|state| self.atom_evaluator.evaluate(state, atom))
            .collect()
    }

    fn eval_nary_and(&self, children: &[CtlFormula<Atom>]) -> Vec<bool> {
        let Some((first, rest)) = children.split_first() else {
            return vec![true; self.state_count()];
        };

        let mut result = self.eval(first);
        for child in rest {
            let sat = self.eval(child);
            for (value, child_sat) in result.iter_mut().zip(sat) {
                *value &= child_sat;
            }
            if result.iter().all(|&value| !value) {
                break;
            }
        }
        result
    }

    fn eval_nary_or(&self, children: &[CtlFormula<Atom>]) -> Vec<bool> {
        let Some((first, rest)) = children.split_first() else {
            return vec![false; self.state_count()];
        };

        let mut result = self.eval(first);
        for child in rest {
            let sat = self.eval(child);
            for (value, child_sat) in result.iter_mut().zip(sat) {
                *value |= child_sat;
            }
            if result.iter().all(|&value| value) {
                break;
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// TransitionSystem bridge
// ---------------------------------------------------------------------------

/// Adapter that bridges [`AtomEvaluator<TS>`] to [`CtlAtomEvaluator`].
///
/// This allows callers that already have a `TransitionSystem` + `AtomEvaluator`
/// pair to use the CTL engine without creating a custom `CtlAtomEvaluator`.
/// Atoms are represented as `usize` indices passed to
/// [`AtomEvaluator::evaluate`].
struct AtomEvalBridge<'a, TS: TransitionSystem> {
    inner: &'a dyn AtomEvaluator<TS>,
}

impl<TS: TransitionSystem> CtlAtomEvaluator<TS::State, usize> for AtomEvalBridge<'_, TS> {
    fn evaluate(&self, state: &TS::State, atom: &usize) -> bool {
        self.inner.evaluate(state, *atom)
    }
}

/// Evaluate a CTL formula over an explicit state graph using the
/// [`TransitionSystem`] / [`AtomEvaluator`] trait pair.
///
/// This is a convenience entry point that builds a [`CtlEngine`] internally.
/// Atoms in the formula are `usize` identifiers resolved by `atom_eval`.
///
/// # Arguments
///
/// - `states` -- all reachable states, indexed `0..N`.
/// - `successors` -- `successors[i]` lists successor state indices for state
///   `i`.
/// - `predecessors` -- `predecessors[i]` lists predecessor state indices for
///   state `i`.
/// - `formula` -- the CTL formula to evaluate.
/// - `atom_eval` -- evaluator for atomic propositions.
///
/// # Returns
///
/// A `Vec<bool>` of length `N` where entry `i` is `true` iff `states[i]`
/// satisfies the formula.
#[must_use]
pub fn check_ctl<TS: TransitionSystem>(
    states: &[TS::State],
    successors: &[Vec<u32>],
    predecessors: &[Vec<u32>],
    formula: &CtlFormula<usize>,
    atom_eval: &dyn AtomEvaluator<TS>,
) -> Vec<bool> {
    let graph = IndexedCtlGraph::new(states, successors, predecessors);
    let bridge = AtomEvalBridge::<TS> { inner: atom_eval };
    let engine = CtlEngine::new(graph, bridge);
    engine.eval(formula)
}

#[cfg(test)]
mod tests {
    use super::{
        build_predecessor_adjacency, CtlAtomEvaluator, CtlEdge, CtlEngine, CtlFormula,
        IndexedCtlGraph,
    };

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestState(Vec<u8>);

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AtLeast {
        index: usize,
        value: u8,
    }

    #[derive(Clone, Copy)]
    struct StateAtomEval;

    impl CtlAtomEvaluator<TestState, AtLeast> for StateAtomEval {
        fn evaluate(&self, state: &TestState, atom: &AtLeast) -> bool {
            state.0[atom.index] >= atom.value
        }
    }

    fn atom(index: usize, value: u8) -> CtlFormula<AtLeast> {
        CtlFormula::Atom(AtLeast { index, value })
    }

    fn engine(
        successors: Vec<Vec<(u32, &'static str)>>,
        states: Vec<TestState>,
    ) -> CtlEngine<'static, TestState, (u32, &'static str), AtLeast, StateAtomEval> {
        let successors = Box::leak(Box::new(successors));
        let predecessors = Box::leak(Box::new(build_predecessor_adjacency(successors)));
        let states = Box::leak(Box::new(states));
        let graph = IndexedCtlGraph::new(states, successors, predecessors);
        CtlEngine::new(graph, StateAtomEval)
    }

    #[test]
    fn predecessor_adjacency_accepts_labeled_edges() {
        let predecessors =
            build_predecessor_adjacency(&[vec![(1, "a"), (2, "b")], vec![(2, "c")], vec![]]);
        assert_eq!(predecessors, vec![vec![], vec![0], vec![0, 1]]);
    }

    #[test]
    fn deadlock_semantics_match_mcc_rules() {
        let engine = engine(vec![vec![]], vec![TestState(vec![1])]);

        assert_eq!(
            engine.eval(&CtlFormula::EX(Box::new(atom(0, 1)))),
            vec![false]
        );
        assert_eq!(
            engine.eval(&CtlFormula::AX(Box::new(atom(0, 1)))),
            vec![true]
        );
        assert_eq!(
            engine.eval(&CtlFormula::EG(Box::new(atom(0, 1)))),
            vec![true]
        );
        assert_eq!(
            engine.eval(&CtlFormula::AG(Box::new(atom(0, 1)))),
            vec![true]
        );

        assert_eq!(
            engine.eval(&CtlFormula::EX(Box::new(atom(0, 2)))),
            vec![false]
        );
        assert_eq!(
            engine.eval(&CtlFormula::AX(Box::new(atom(0, 2)))),
            vec![true]
        );
        assert_eq!(
            engine.eval(&CtlFormula::EG(Box::new(atom(0, 2)))),
            vec![false]
        );
        assert_eq!(
            engine.eval(&CtlFormula::AG(Box::new(atom(0, 2)))),
            vec![false]
        );
    }

    #[test]
    fn until_and_fixpoints_follow_existing_semantics() {
        let engine = engine(
            vec![
                vec![(1, "a"), (2, "b")],
                vec![(3, "c")],
                vec![(2, "d")],
                vec![(3, "e")],
            ],
            vec![
                TestState(vec![1, 0]),
                TestState(vec![1, 1]),
                TestState(vec![0, 0]),
                TestState(vec![0, 1]),
            ],
        );

        let phi = atom(0, 1);
        let psi = atom(1, 1);
        let direct = CtlFormula::AU(Box::new(phi.clone()), Box::new(psi.clone()));
        let rewritten = CtlFormula::Not(Box::new(CtlFormula::Or(vec![
            CtlFormula::EU(
                Box::new(CtlFormula::Not(Box::new(psi.clone()))),
                Box::new(CtlFormula::And(vec![
                    CtlFormula::Not(Box::new(phi.clone())),
                    CtlFormula::Not(Box::new(psi.clone())),
                ])),
            ),
            CtlFormula::EG(Box::new(CtlFormula::Not(Box::new(psi.clone())))),
        ])));

        assert_eq!(engine.eval(&direct), engine.eval(&rewritten));
        assert_eq!(engine.eval(&direct), vec![false, true, false, true]);
    }

    #[test]
    fn edge_trait_supports_plain_successor_ids() {
        assert_eq!(7u32.successor(), 7);
    }

    // -----------------------------------------------------------------------
    // Comprehensive per-operator tests on small graphs
    // -----------------------------------------------------------------------

    // Helper: build engine from plain u32 adjacency (no labels).
    fn engine_plain(
        adj: Vec<Vec<u32>>,
        states: Vec<TestState>,
    ) -> CtlEngine<'static, TestState, u32, AtLeast, StateAtomEval> {
        let adj = Box::leak(Box::new(adj));
        let predecessors = Box::leak(Box::new(build_predecessor_adjacency::<u32>(adj)));
        let states = Box::leak(Box::new(states));
        let graph = IndexedCtlGraph::new(states, adj, predecessors);
        CtlEngine::new(graph, StateAtomEval)
    }

    // ---- Atom / Not / And / Or ----

    #[test]
    fn atom_evaluation_basic() {
        // 3-state linear: 0->1->2(deadlock), values [0], [1], [2]
        let e = engine_plain(
            vec![vec![1], vec![2], vec![]],
            vec![TestState(vec![0]), TestState(vec![1]), TestState(vec![2])],
        );
        // atom(0,1): state.0[0] >= 1 => {1,2}
        assert_eq!(e.eval(&atom(0, 1)), vec![false, true, true]);
        // atom(0,2): state.0[0] >= 2 => {2}
        assert_eq!(e.eval(&atom(0, 2)), vec![false, false, true]);
    }

    #[test]
    fn not_inverts_satisfaction() {
        let e = engine_plain(
            vec![vec![1], vec![]],
            vec![TestState(vec![0]), TestState(vec![1])],
        );
        assert_eq!(
            e.eval(&CtlFormula::Not(Box::new(atom(0, 1)))),
            vec![true, false]
        );
    }

    #[test]
    fn and_intersection() {
        // 3 states with two-element vectors
        let e = engine_plain(
            vec![vec![1], vec![2], vec![]],
            vec![
                TestState(vec![1, 0]),
                TestState(vec![0, 1]),
                TestState(vec![1, 1]),
            ],
        );
        // atom(0,1) AND atom(1,1): only state 2 has both >= 1
        let f = CtlFormula::And(vec![atom(0, 1), atom(1, 1)]);
        assert_eq!(e.eval(&f), vec![false, false, true]);
    }

    #[test]
    fn or_union() {
        let e = engine_plain(
            vec![vec![1], vec![2], vec![]],
            vec![
                TestState(vec![1, 0]),
                TestState(vec![0, 1]),
                TestState(vec![0, 0]),
            ],
        );
        // atom(0,1) OR atom(1,1): states 0 and 1
        let f = CtlFormula::Or(vec![atom(0, 1), atom(1, 1)]);
        assert_eq!(e.eval(&f), vec![true, true, false]);
    }

    #[test]
    fn and_empty_is_true_everywhere() {
        let e = engine_plain(vec![vec![0]], vec![TestState(vec![0])]);
        assert_eq!(e.eval(&CtlFormula::And(vec![])), vec![true]);
    }

    #[test]
    fn or_empty_is_false_everywhere() {
        let e = engine_plain(vec![vec![0]], vec![TestState(vec![0])]);
        assert_eq!(e.eval(&CtlFormula::Or(vec![])), vec![false]);
    }

    // ---- EX ----

    #[test]
    fn ex_some_successor_satisfies() {
        // 0->{1,2}, 1->1, 2(deadlock). Values: [0], [1], [0].
        let e = engine_plain(
            vec![vec![1, 2], vec![1], vec![]],
            vec![TestState(vec![0]), TestState(vec![1]), TestState(vec![0])],
        );
        // EX(atom(0,1)): successor with val >= 1
        // State 0: succs {1,2}. State 1 has [1] => true
        // State 1: succs {1}. Self-loop [1] => true
        // State 2: deadlock => false
        assert_eq!(
            e.eval(&CtlFormula::EX(Box::new(atom(0, 1)))),
            vec![true, true, false]
        );
    }

    #[test]
    fn ex_at_deadlock_is_false() {
        let e = engine_plain(vec![vec![]], vec![TestState(vec![1])]);
        assert_eq!(e.eval(&CtlFormula::EX(Box::new(atom(0, 1)))), vec![false]);
    }

    // ---- AX ----

    #[test]
    fn ax_all_successors_satisfy() {
        // 0->{1,2}, 1->1, 2(deadlock). Values: [0], [1], [0].
        let e = engine_plain(
            vec![vec![1, 2], vec![1], vec![]],
            vec![TestState(vec![0]), TestState(vec![1]), TestState(vec![0])],
        );
        // AX(atom(0,1)):
        // State 0: succs {1,2}; state 2 is [0] => not all => false
        // State 1: succs {1}; [1] => true
        // State 2: deadlock => vacuously true
        assert_eq!(
            e.eval(&CtlFormula::AX(Box::new(atom(0, 1)))),
            vec![false, true, true]
        );
    }

    #[test]
    fn ax_ex_duality() {
        // AX(phi) == NOT EX(NOT phi) on non-deadlock states; at deadlock both
        // AX=true and NOT EX(NOT phi)=NOT false=true, so duality holds.
        let e = engine_plain(
            vec![vec![1, 2], vec![1], vec![2]],
            vec![TestState(vec![0]), TestState(vec![1]), TestState(vec![0])],
        );
        let phi = atom(0, 1);
        let ax_sat = e.eval(&CtlFormula::AX(Box::new(phi.clone())));
        let dual = e.eval(&CtlFormula::Not(Box::new(CtlFormula::EX(Box::new(
            CtlFormula::Not(Box::new(phi)),
        )))));
        assert_eq!(ax_sat, dual);
    }

    // ---- EF ----

    #[test]
    fn ef_backward_reachability() {
        // 0->1->2(deadlock). Atom true only at state 2.
        let e = engine_plain(
            vec![vec![1], vec![2], vec![]],
            vec![TestState(vec![0]), TestState(vec![0]), TestState(vec![1])],
        );
        // EF(atom(0,1)): can reach state 2?
        // 0->1->2 => all can reach => [true, true, true]
        assert_eq!(
            e.eval(&CtlFormula::EF(Box::new(atom(0, 1)))),
            vec![true, true, true]
        );
    }

    #[test]
    fn ef_unreachable() {
        // 0->1->1(self-loop), 2(deadlock isolated). Atom true only at 2.
        // But 0 and 1 cannot reach 2.
        let e = engine_plain(
            vec![vec![1], vec![1], vec![]],
            vec![TestState(vec![0]), TestState(vec![0]), TestState(vec![1])],
        );
        // EF(atom(0,1)):
        // State 2: atom true => EF true
        // States 0,1: can only reach each other => EF false
        assert_eq!(
            e.eval(&CtlFormula::EF(Box::new(atom(0, 1)))),
            vec![false, false, true]
        );
    }

    // ---- AF ----

    #[test]
    fn af_all_paths_reach() {
        // 0->{1,2}, 1->3, 2->3, 3(deadlock). Atom true at 3.
        // All paths from any state reach 3.
        let e = engine_plain(
            vec![vec![1, 2], vec![3], vec![3], vec![]],
            vec![
                TestState(vec![0]),
                TestState(vec![0]),
                TestState(vec![0]),
                TestState(vec![1]),
            ],
        );
        assert_eq!(
            e.eval(&CtlFormula::AF(Box::new(atom(0, 1)))),
            vec![true, true, true, true]
        );
    }

    #[test]
    fn af_some_path_avoids() {
        // 0->{1,2}, 1->1(self-loop), 2(deadlock). Atom true at 2.
        // Path 0->1->1->... never reaches 2. AF = false at 0 and 1.
        // But wait, AF means "on ALL paths, eventually". State 0 has path
        // 0->1->1->... which never reaches atom. But atom holds at 0? No, atom
        // is true only at 2. So AF(atom) at 0: path 0->2 reaches it, but path
        // 0->1->1->... doesn't. => false.
        let e = engine_plain(
            vec![vec![1, 2], vec![1], vec![]],
            vec![TestState(vec![0]), TestState(vec![0]), TestState(vec![1])],
        );
        // AF at 0: false (path 0->1->... avoids)
        // AF at 1: false (self-loop, never reaches)
        // AF at 2: true (holds here)
        assert_eq!(
            e.eval(&CtlFormula::AF(Box::new(atom(0, 1)))),
            vec![false, false, true]
        );
    }

    #[test]
    fn af_eg_duality() {
        // AF(phi) == NOT EG(NOT phi)
        let e = engine_plain(
            vec![vec![1, 2], vec![1], vec![2]],
            vec![TestState(vec![0]), TestState(vec![1]), TestState(vec![0])],
        );
        let phi = atom(0, 1);
        let af_sat = e.eval(&CtlFormula::AF(Box::new(phi.clone())));
        let dual = e.eval(&CtlFormula::Not(Box::new(CtlFormula::EG(Box::new(
            CtlFormula::Not(Box::new(phi)),
        )))));
        assert_eq!(af_sat, dual);
    }

    // ---- EG ----

    #[test]
    fn eg_cycle_keeps_all() {
        // Cycle: 0->1->2->0. All states satisfy atom.
        let e = engine_plain(
            vec![vec![1], vec![2], vec![0]],
            vec![TestState(vec![1]), TestState(vec![1]), TestState(vec![1])],
        );
        // EG(atom): infinite path cycling => all true
        assert_eq!(
            e.eval(&CtlFormula::EG(Box::new(atom(0, 1)))),
            vec![true, true, true]
        );
    }

    #[test]
    fn eg_breaks_at_non_satisfying_successor() {
        // 0->1->2(deadlock). All satisfy atom but 2 is deadlock with atom=false.
        // Wait, need to be careful with maximal-path semantics.
        // Let's have: 0->1->2(deadlock), atom true at {0,1}, false at {2}.
        // EG: 2 is deadlock, atom=false => not in set.
        // 1: only succ=2, not in set => removed.
        // 0: only succ=1, not in set => removed.
        let e = engine_plain(
            vec![vec![1], vec![2], vec![]],
            vec![TestState(vec![1]), TestState(vec![1]), TestState(vec![0])],
        );
        assert_eq!(
            e.eval(&CtlFormula::EG(Box::new(atom(0, 1)))),
            vec![false, false, false]
        );
    }

    #[test]
    fn eg_deadlock_maximal_path_semantics() {
        // Single deadlock state, atom=true => stays (path=[s]).
        let e = engine_plain(vec![vec![]], vec![TestState(vec![1])]);
        assert_eq!(e.eval(&CtlFormula::EG(Box::new(atom(0, 1)))), vec![true]);
        // atom=false => gone
        let e2 = engine_plain(vec![vec![]], vec![TestState(vec![0])]);
        assert_eq!(e2.eval(&CtlFormula::EG(Box::new(atom(0, 1)))), vec![false]);
    }

    #[test]
    fn eg_branch_to_cycle_and_deadlock() {
        // 0->{1,2}, 1->1 (self-loop), 2(deadlock).
        // atom true at {0,1,2}.
        // EG(atom):
        //   State 2: deadlock, atom=true => stays (maximal path).
        //   State 1: self-loop, atom=true => stays.
        //   State 0: succ {1,2}, both in set => stays.
        let e = engine_plain(
            vec![vec![1, 2], vec![1], vec![]],
            vec![TestState(vec![1]), TestState(vec![1]), TestState(vec![1])],
        );
        assert_eq!(
            e.eval(&CtlFormula::EG(Box::new(atom(0, 1)))),
            vec![true, true, true]
        );
    }

    // ---- AG ----

    #[test]
    fn ag_globally_true() {
        // Cycle: all satisfy atom => AG true everywhere
        let e = engine_plain(
            vec![vec![1], vec![0]],
            vec![TestState(vec![1]), TestState(vec![1])],
        );
        assert_eq!(
            e.eval(&CtlFormula::AG(Box::new(atom(0, 1)))),
            vec![true, true]
        );
    }

    #[test]
    fn ag_reachable_violation() {
        // 0->1->2(deadlock). Atom false at 2.
        // AG(atom): from 0 you reach 2 which violates => false at {0,1}.
        let e = engine_plain(
            vec![vec![1], vec![2], vec![]],
            vec![TestState(vec![1]), TestState(vec![1]), TestState(vec![0])],
        );
        assert_eq!(
            e.eval(&CtlFormula::AG(Box::new(atom(0, 1)))),
            vec![false, false, false]
        );
    }

    #[test]
    fn ag_ef_not_duality() {
        // AG(phi) == NOT EF(NOT phi)
        let e = engine_plain(
            vec![vec![1, 2], vec![1], vec![]],
            vec![TestState(vec![1]), TestState(vec![1]), TestState(vec![0])],
        );
        let phi = atom(0, 1);
        let ag = e.eval(&CtlFormula::AG(Box::new(phi.clone())));
        let dual = e.eval(&CtlFormula::Not(Box::new(CtlFormula::EF(Box::new(
            CtlFormula::Not(Box::new(phi)),
        )))));
        assert_eq!(ag, dual);
    }

    // ---- EU ----

    #[test]
    fn eu_phi_until_psi_basic() {
        // 0->1->2(deadlock). States: [1,0], [1,0], [0,1].
        // EU(atom(0,1), atom(1,1)): phi holds at {0,1}, psi at {2}.
        // Backward from 2: pred=1, phi[1]=true => add; pred of 1=0, phi[0]=true => add.
        let e = engine_plain(
            vec![vec![1], vec![2], vec![]],
            vec![
                TestState(vec![1, 0]),
                TestState(vec![1, 0]),
                TestState(vec![0, 1]),
            ],
        );
        assert_eq!(
            e.eval(&CtlFormula::EU(Box::new(atom(0, 1)), Box::new(atom(1, 1)))),
            vec![true, true, true]
        );
    }

    #[test]
    fn eu_phi_breaks_midway() {
        // 0->1->2(deadlock). States: [1,0], [0,0], [0,1].
        // phi at {0}, psi at {2}. But state 1 has phi=false => can't propagate.
        let e = engine_plain(
            vec![vec![1], vec![2], vec![]],
            vec![
                TestState(vec![1, 0]),
                TestState(vec![0, 0]),
                TestState(vec![0, 1]),
            ],
        );
        assert_eq!(
            e.eval(&CtlFormula::EU(Box::new(atom(0, 1)), Box::new(atom(1, 1)))),
            vec![false, false, true]
        );
    }

    #[test]
    fn eu_psi_holds_immediately() {
        // E[phi U psi]: if psi holds at a state, EU is true regardless of phi.
        let e = engine_plain(
            vec![vec![1], vec![]],
            vec![TestState(vec![0, 1]), TestState(vec![0, 1])],
        );
        assert_eq!(
            e.eval(&CtlFormula::EU(
                Box::new(atom(0, 1)), // phi=false
                Box::new(atom(1, 1))  // psi=true
            )),
            vec![true, true]
        );
    }

    // ---- AU ----

    #[test]
    fn au_all_paths_phi_until_psi() {
        // 0->{1,2}, 1->3, 2->3, 3(deadlock). phi at {0,1,2}, psi at {3}.
        // All paths from 0 go through phi-states until reaching psi=3.
        let e = engine_plain(
            vec![vec![1, 2], vec![3], vec![3], vec![]],
            vec![
                TestState(vec![1, 0]),
                TestState(vec![1, 0]),
                TestState(vec![1, 0]),
                TestState(vec![0, 1]),
            ],
        );
        assert_eq!(
            e.eval(&CtlFormula::AU(Box::new(atom(0, 1)), Box::new(atom(1, 1)))),
            vec![true, true, true, true]
        );
    }

    #[test]
    fn au_one_path_escapes() {
        // 0->{1,2}, 1->1(self-loop), 2(deadlock). phi at {0,1}, psi at {2}.
        // Path 0->1->1->... never reaches psi => AU false at 0.
        let e = engine_plain(
            vec![vec![1, 2], vec![1], vec![]],
            vec![
                TestState(vec![1, 0]),
                TestState(vec![1, 0]),
                TestState(vec![0, 1]),
            ],
        );
        // State 0: path 0->1->1->... avoids psi => false
        // State 1: self-loop, never reaches psi => false
        // State 2: psi holds => true
        assert_eq!(
            e.eval(&CtlFormula::AU(Box::new(atom(0, 1)), Box::new(atom(1, 1)))),
            vec![false, false, true]
        );
    }

    // ---- AU exhaustive cross-validation (small deadlock-free graphs) ----

    fn bitvec(mask: u32, n: usize) -> Vec<bool> {
        (0..n).map(|i| mask & (1 << i) != 0).collect()
    }

    #[test]
    fn au_algebraic_vs_fixpoint_exhaustive_deadlock_free() {
        // On graphs without deadlocks, the algebraic identity used by AU
        // matches the direct fixpoint mu Z. psi | (phi & AX(Z)).
        let graphs: &[(&str, &[&[u32]])] = &[
            ("cycle_3", &[&[1], &[2], &[0]]),
            ("single_cycle", &[&[0]]),
            ("two_cycles", &[&[1, 2], &[0], &[2]]),
        ];

        let mut total = 0;
        for &(name, adj) in graphs {
            let n = adj.len();
            let adj_vec: Vec<Vec<u32>> = adj.iter().map(|s| s.to_vec()).collect();
            let states: Vec<TestState> = (0..n).map(|_| TestState(vec![0])).collect();
            let e = engine_plain(adj_vec, states);

            for phi_mask in 0..(1u32 << n) {
                let phi = bitvec(phi_mask, n);
                for psi_mask in 0..(1u32 << n) {
                    let psi = bitvec(psi_mask, n);

                    // Algebraic AU (as implemented)
                    let not_phi: Vec<bool> = phi.iter().map(|&v| !v).collect();
                    let not_psi: Vec<bool> = psi.iter().map(|&v| !v).collect();
                    let both_not: Vec<bool> = (0..n).map(|i| not_phi[i] && not_psi[i]).collect();
                    let eu_r = e.lfp_eu(&not_psi, &both_not);
                    let eg_r = e.gfp_eg(&not_psi);
                    let alg: Vec<bool> = (0..n).map(|i| !(eu_r[i] || eg_r[i])).collect();

                    // Reference fixpoint: mu Z. psi | (phi & AX(Z))
                    let mut z = vec![false; n];
                    loop {
                        let ax_z = e.pre_a(&z);
                        let new_z: Vec<bool> =
                            (0..n).map(|i| psi[i] || (phi[i] && ax_z[i])).collect();
                        if new_z == z {
                            break;
                        }
                        z = new_z;
                    }

                    assert_eq!(alg, z, "AU mismatch on '{name}': phi={phi:?} psi={psi:?}");
                    total += 1;
                }
            }
        }
        eprintln!("AU exhaustive: {total} checks passed");
    }

    // ---- EGF (fair-cycle persistence carrier) ----

    fn egf(inner: CtlFormula<AtLeast>) -> CtlFormula<AtLeast> {
        CtlFormula::EGF(Box::new(inner))
    }

    #[test]
    fn egf_two_cycle_visits_atom_infinitely() {
        // 0<->1 toggle; atom true only at state 0. The cycle visits state 0
        // (the atom) infinitely often, so EGF holds at BOTH states.
        let e = engine_plain(
            vec![vec![1], vec![0]],
            vec![TestState(vec![1]), TestState(vec![0])],
        );
        assert_eq!(e.eval(&egf(atom(0, 1))), vec![true, true]);
    }

    #[test]
    fn egf_self_loop_on_atom_holds() {
        // Single self-looping state satisfying the atom => visited infinitely.
        let e = engine_plain(vec![vec![0]], vec![TestState(vec![1])]);
        assert_eq!(e.eval(&egf(atom(0, 1))), vec![true]);
    }

    #[test]
    fn egf_chain_leaving_atom_is_false() {
        // 0->1->2(deadlock); atom true only at state 0. No path returns to 0,
        // and 0 is not a deadlock, so no path visits the atom infinitely often.
        let e = engine_plain(
            vec![vec![1], vec![2], vec![]],
            vec![TestState(vec![1]), TestState(vec![0]), TestState(vec![0])],
        );
        assert_eq!(e.eval(&egf(atom(0, 1))), vec![false, false, false]);
    }

    #[test]
    fn egf_deadlock_stutter_is_a_fair_witness() {
        // 0->1, 1 is a deadlock satisfying the atom. The stutter convention
        // makes the deadlocked atom-state an infinite atom-stutter, so EGF holds
        // at state 1 (and at 0, which reaches it). Without the deadlock-stutter
        // term both would be false — this pins parity with the GPU CtlOp::EGF.
        let e = engine_plain(
            vec![vec![1], vec![]],
            vec![TestState(vec![0]), TestState(vec![1])],
        );
        assert_eq!(e.eval(&egf(atom(0, 1))), vec![true, true]);
    }

    #[test]
    fn egf_deadlock_without_atom_is_not_a_witness() {
        // Same shape but the deadlock does NOT satisfy the atom: no fair path.
        let e = engine_plain(
            vec![vec![1], vec![]],
            vec![TestState(vec![1]), TestState(vec![0])],
        );
        assert_eq!(e.eval(&egf(atom(0, 1))), vec![false, false]);
    }

    #[test]
    fn persistence_dual_afg_is_not_egf_of_negation() {
        // A(FG p) == NOT EGF(NOT p). On the 0<->1 toggle with p true only at 0,
        // the cycle visits NOT p (state 1) infinitely, so A(FG p) is FALSE at
        // both states — matching the fair-cycle reading.
        let e = engine_plain(
            vec![vec![1], vec![0]],
            vec![TestState(vec![1]), TestState(vec![0])],
        );
        let afg = CtlFormula::Not(Box::new(egf(CtlFormula::Not(Box::new(atom(0, 1))))));
        assert_eq!(e.eval(&afg), vec![false, false]);

        // On a stuck-at-p deadlock (0 is a deadlock satisfying p), A(FG p) holds.
        let e2 = engine_plain(vec![vec![]], vec![TestState(vec![1])]);
        let afg2 = CtlFormula::Not(Box::new(egf(CtlFormula::Not(Box::new(atom(0, 1))))));
        assert_eq!(e2.eval(&afg2), vec![true]);
    }

    #[test]
    fn egf_eval_root_matches_full_eval() {
        // eval_root must be verdict-identical to eval(..)[0] for EGF.
        let e = engine_plain(
            vec![vec![1], vec![0]],
            vec![TestState(vec![1]), TestState(vec![0])],
        );
        let f = egf(atom(0, 1));
        assert_eq!(e.eval_root(&f), e.eval(&f)[0]);
        let g = CtlFormula::Not(Box::new(egf(CtlFormula::Not(Box::new(atom(0, 1))))));
        assert_eq!(e.eval_root(&g), e.eval(&g)[0]);
    }

    // ---- eval_root differential: early-exit root evaluation must be
    // verdict-identical to the full fixpoint at state 0 ----

    #[test]
    fn eval_root_matches_eval_index0_exhaustive() {
        // Graph shapes chosen to cover: deadlocks, self-loops, cycles,
        // branching into cycle+deadlock, and the initial state being
        // decided early vs last in each fixpoint direction.
        let graphs: &[(&str, &[&[u32]])] = &[
            ("deadlock_single", &[&[]]),
            ("self_loop", &[&[0]]),
            ("linear_to_deadlock", &[&[1], &[2], &[]]),
            ("cycle_3", &[&[1], &[2], &[0]]),
            ("branch_cycle_deadlock", &[&[1, 2], &[1], &[]]),
            ("diamond", &[&[1, 2], &[3], &[3], &[]]),
            ("two_cycles", &[&[1, 2], &[0], &[2]]),
        ];

        let mut total = 0usize;
        for &(name, adj) in graphs {
            let n = adj.len();
            let adj_vec: Vec<Vec<u32>> = adj.iter().map(|s| s.to_vec()).collect();

            for phi_mask in 0..(1u32 << n) {
                for psi_mask in 0..(1u32 << n) {
                    // Encode the two atom masks into the state payload.
                    let states: Vec<TestState> = (0..n)
                        .map(|i| {
                            TestState(vec![
                                u8::from(phi_mask & (1 << i) != 0),
                                u8::from(psi_mask & (1 << i) != 0),
                            ])
                        })
                        .collect();
                    let e = engine_plain(adj_vec.clone(), states);
                    let phi = || atom(0, 1);
                    let psi = || atom(1, 1);

                    let formulas: Vec<CtlFormula<AtLeast>> = vec![
                        phi(),
                        CtlFormula::Not(Box::new(phi())),
                        CtlFormula::And(vec![phi(), psi()]),
                        CtlFormula::Or(vec![phi(), psi()]),
                        CtlFormula::EX(Box::new(phi())),
                        CtlFormula::AX(Box::new(phi())),
                        CtlFormula::EF(Box::new(phi())),
                        CtlFormula::AF(Box::new(phi())),
                        CtlFormula::EG(Box::new(phi())),
                        CtlFormula::AG(Box::new(phi())),
                        CtlFormula::EU(Box::new(phi()), Box::new(psi())),
                        CtlFormula::AU(Box::new(phi()), Box::new(psi())),
                        CtlFormula::Not(Box::new(CtlFormula::EF(Box::new(phi())))),
                        CtlFormula::Not(Box::new(CtlFormula::EG(Box::new(phi())))),
                        CtlFormula::AG(Box::new(CtlFormula::EF(Box::new(phi())))),
                        CtlFormula::AF(Box::new(CtlFormula::EG(Box::new(phi())))),
                        CtlFormula::EU(
                            Box::new(CtlFormula::Not(Box::new(psi()))),
                            Box::new(CtlFormula::And(vec![phi(), psi()])),
                        ),
                        CtlFormula::AU(
                            Box::new(CtlFormula::Or(vec![phi(), psi()])),
                            Box::new(CtlFormula::Not(Box::new(phi()))),
                        ),
                    ];

                    for (idx, formula) in formulas.iter().enumerate() {
                        let full = e.eval(formula)[0];
                        let root = e.eval_root(formula);
                        assert_eq!(
                            root, full,
                            "eval_root mismatch on '{name}' formula #{idx} \
                             phi={phi_mask:#b} psi={psi_mask:#b}: root={root} full={full}"
                        );
                        total += 1;
                    }
                }
            }
        }
        eprintln!("eval_root differential: {total} checks, 0 disagreements");
    }

    // ---- Nested formulas ----

    #[test]
    fn nested_ag_ef() {
        // 0->{1,2}, 1->1, 2(deadlock). Atom true at {0,1}.
        let e = engine_plain(
            vec![vec![1, 2], vec![1], vec![]],
            vec![TestState(vec![1]), TestState(vec![1]), TestState(vec![0])],
        );
        // EF(atom) = [true, true, false]
        // AG(EF(atom)): state 0 can reach 2 (EF=false) => AG false at 0
        assert_eq!(
            e.eval(&CtlFormula::AG(Box::new(CtlFormula::EF(Box::new(atom(
                0, 1
            )))))),
            vec![false, true, false]
        );
    }

    #[test]
    fn nested_af_eg() {
        // Same graph.
        let e = engine_plain(
            vec![vec![1, 2], vec![1], vec![]],
            vec![TestState(vec![1]), TestState(vec![1]), TestState(vec![0])],
        );
        // EG(atom) = [true, true, false]
        // AF(EG(atom)): state 0 has EG true => AF true
        assert_eq!(
            e.eval(&CtlFormula::AF(Box::new(CtlFormula::EG(Box::new(atom(
                0, 1
            )))))),
            vec![true, true, false]
        );
    }

    // ---- check_ctl bridge ----

    #[test]
    fn check_ctl_bridge_basic() {
        use crate::{AtomEvaluator, TransitionSystem};

        #[derive(Clone)]
        struct TinyTS;

        impl TransitionSystem for TinyTS {
            type State = u8;
            type Action = ();
            type Fingerprint = u8;

            fn initial_states(&self) -> Vec<u8> {
                vec![0]
            }
            fn successors(&self, _: &u8) -> Vec<((), u8)> {
                Vec::new()
            }
            fn fingerprint(&self, s: &u8) -> u8 {
                *s
            }
        }

        struct TinyEval;
        impl AtomEvaluator<TinyTS> for TinyEval {
            fn evaluate(&self, state: &u8, atom_id: usize) -> bool {
                match atom_id {
                    0 => *state >= 1,
                    _ => false,
                }
            }
        }

        let states: Vec<u8> = vec![0, 1, 2];
        let successors = vec![vec![1u32], vec![2], vec![]];
        let predecessors = build_predecessor_adjacency(&successors);

        // EF(atom 0): state[i] >= 1 at {1,2}. backward: 2<-1<-0.
        let result = super::check_ctl::<TinyTS>(
            &states,
            &successors,
            &predecessors,
            &CtlFormula::EF(Box::new(CtlFormula::Atom(0))),
            &TinyEval,
        );
        assert_eq!(result, vec![true, true, true]);

        // AG(atom 0): state 0 has val=0, atom false => AG false at all
        // reaching 0. State 0 is initial so AG false at 0.
        let result = super::check_ctl::<TinyTS>(
            &states,
            &successors,
            &predecessors,
            &CtlFormula::AG(Box::new(CtlFormula::Atom(0))),
            &TinyEval,
        );
        // Not all reachable states satisfy: state 0 fails.
        // From state 0: atom=false => AG false.
        // From state 1: reaches 2(deadlock) atom=true, and state 1 atom=true => AG true.
        // From state 2: deadlock, atom=true => AG true.
        assert_eq!(result, vec![false, true, true]);
    }

    // ---- Empty graph ----

    #[test]
    fn empty_graph_returns_empty() {
        let e = engine_plain(vec![], vec![]);
        assert_eq!(e.eval(&CtlFormula::And(vec![])), Vec::<bool>::new());
        assert_eq!(
            e.eval(&CtlFormula::EG(Box::new(atom(0, 1)))),
            Vec::<bool>::new()
        );
    }

    // ---- CSR predecessor adjacency (audit S4/S6) ----

    /// The CSR builder must produce EXACTLY the nested builder's per-state
    /// predecessor lists — contents and order — on every shape: empty,
    /// deadlocks, self-loops, parallel (duplicate) edges, dense cycles.
    #[test]
    fn csr_predecessors_match_nested_exactly() {
        let graphs: &[&[&[u32]]] = &[
            &[],
            &[&[]],
            &[&[0]],
            &[&[1], &[2], &[]],
            &[&[1, 2], &[1], &[]],
            &[&[1], &[2], &[0]],
            // Parallel edges: 0 -> 1 twice, plus 1 -> 0.
            &[&[1, 1], &[0]],
            // Dense: every state to every state (self-loops included).
            &[&[0, 1, 2], &[0, 1, 2], &[0, 1, 2]],
            // Skewed in-degree: everything converges on state 3.
            &[&[3], &[3], &[3], &[]],
        ];
        for &adj in graphs {
            let adj_vec: Vec<Vec<u32>> = adj.iter().map(|s| s.to_vec()).collect();
            let nested = build_predecessor_adjacency(&adj_vec);
            let csr = super::build_predecessor_csr(&adj_vec);
            assert_eq!(csr.state_count(), nested.len());
            for (s, expected) in nested.iter().enumerate() {
                assert_eq!(
                    csr.neighbors(s),
                    expected.as_slice(),
                    "CSR predecessor list drift at state {s} for {adj_vec:?}"
                );
            }
        }
    }

    /// Engine verdicts over the CSR predecessor view must be identical to
    /// the nested view on the full operator battery (the same graphs and
    /// formula shapes as `eval_root_matches_eval_index0_exhaustive`).
    #[test]
    fn csr_engine_matches_nested_engine_exhaustive() {
        let graphs: &[&[&[u32]]] = &[
            &[&[]],
            &[&[0]],
            &[&[1], &[2], &[]],
            &[&[1], &[2], &[0]],
            &[&[1, 2], &[1], &[]],
            &[&[1, 2], &[3], &[3], &[]],
            &[&[1, 2], &[0], &[2]],
        ];

        for &adj in graphs {
            let n = adj.len();
            let adj_vec: Vec<Vec<u32>> = adj.iter().map(|s| s.to_vec()).collect();
            for phi_mask in 0..(1u32 << n) {
                for psi_mask in 0..(1u32 << n) {
                    let states: Vec<TestState> = (0..n)
                        .map(|i| {
                            TestState(vec![
                                u8::from(phi_mask & (1 << i) != 0),
                                u8::from(psi_mask & (1 << i) != 0),
                            ])
                        })
                        .collect();

                    let nested = build_predecessor_adjacency::<u32>(&adj_vec);
                    let csr = super::build_predecessor_csr::<u32>(&adj_vec);
                    let e_nested = CtlEngine::new(
                        IndexedCtlGraph::new(&states, &adj_vec, &nested),
                        StateAtomEval,
                    );
                    let e_csr = CtlEngine::new(
                        IndexedCtlGraph::new_with_csr_predecessors(&states, &adj_vec, &csr),
                        StateAtomEval,
                    );

                    let phi = || atom(0, 1);
                    let psi = || atom(1, 1);
                    let formulas: Vec<CtlFormula<AtLeast>> = vec![
                        CtlFormula::EX(Box::new(phi())),
                        CtlFormula::AX(Box::new(phi())),
                        CtlFormula::EF(Box::new(phi())),
                        CtlFormula::AF(Box::new(phi())),
                        CtlFormula::EG(Box::new(phi())),
                        CtlFormula::AG(Box::new(phi())),
                        CtlFormula::EU(Box::new(phi()), Box::new(psi())),
                        CtlFormula::AU(Box::new(phi()), Box::new(psi())),
                        CtlFormula::AG(Box::new(CtlFormula::EF(Box::new(phi())))),
                        CtlFormula::AF(Box::new(CtlFormula::EG(Box::new(phi())))),
                    ];
                    for (idx, formula) in formulas.iter().enumerate() {
                        assert_eq!(
                            e_nested.eval(formula),
                            e_csr.eval(formula),
                            "CSR eval drift on formula #{idx} adj={adj_vec:?} \
                             phi={phi_mask:#b} psi={psi_mask:#b}"
                        );
                        assert_eq!(
                            e_nested.eval_root(formula),
                            e_csr.eval_root(formula),
                            "CSR eval_root drift on formula #{idx} adj={adj_vec:?}"
                        );
                    }
                }
            }
        }
    }
}
