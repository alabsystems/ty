// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Unified alternation-free modal mu-calculus solver over Petri-net
//! state spaces.
//!
//! ## Architecture
//!
//! This module is the shared fixpoint engine that the temporal-logic
//! examinations in `tla-petri` build on. It generalises the
//! Liu-Smolka extended dependency graph (EDG) algorithm previously
//! specialised to CTL (see the now-thin
//! `examinations::ctl::local_edg`) into one solver parameterised over
//! a modal mu-calculus formula AST.
//!
//! What is parameterised vs hardcoded:
//!
//! - Parameterised: the [`MuFormula`] AST. New temporal logics (LTL
//!   via alternating-automaton encoding, Liveness, StableMarking) plug
//!   in by translating their formulas into [`MuFormula`] and calling
//!   [`LocalMuSolver::solve`]. The solver does not know about CTL,
//!   LTL, etc.
//! - Hardcoded: the underlying labelled transition system. The current
//!   instantiation is a Petri net with unlabelled `successors_of`
//!   semantics (any transition counts). Action labels can be added
//!   later by widening [`MuFormula::Diamond`] / [`MuFormula::Box`] to
//!   carry an `Action` payload; the solver's dispatch is already
//!   structured so the change would be local.
//!
//! ## Algorithm
//!
//! Standard local Liu-Smolka EDG on `(state, subformula)` vertices.
//! Each operator installs hyperedges and the worklist resolves marks
//! to `True` / `False` / `Unknown`, propagating certain-True and
//! certain-False (certain-zero) both ways and terminating the moment
//! the root is decided either way. When the worklist drains, any
//! still-`Unknown` vertex takes the polarity-default of the innermost
//! enclosing fixpoint, one fixpoint component at a time
//! (innermost-first — see [`LocalMuSolver::fixpoint_close`]):
//!
//! - μ (least fixpoint) defaults to `False` — no witness was ever
//!   produced.
//! - ν (greatest fixpoint) defaults to `True` — no demote-to-False
//!   evidence was ever produced (Dalsgaard et al. 2018
//!   certain-zero / tentative-True closure).
//!
//! This is sound for the *alternation-free* fragment, which is
//! sufficient for CTL, LTL (via alternating automata), Liveness, and
//! StableMarking. Strict alternation (μν or νμ nesting whose inner
//! variable is referenced from the outer) requires a more elaborate
//! progress measure (Jurdzinski small-progress / parity-game
//! algorithm); the solver does not currently attempt it and aborts
//! with [`MuAbort::UnsupportedAlternation`] when detected.
//!
//! ## Soundness gates
//!
//! Every abort path produces an `Err` — no abort can mark the root
//! `True` or `False`. The pipeline maps `Err` to
//! `Verdict::CannotCompute`. Soundness of the supported operators is
//! validated by:
//!
//! - The unit / differential tests in
//!   [`mu_calculus_tests`](super::mu_calculus_tests), which compare
//!   the unified solver against the full-graph
//!   `tla-mc-core::ctl::CtlEngine` on a battery of CTL formulas
//!   translated via [`ctl_to_mu`].
//! - The retained 100 CTL pipeline tests under `examinations::ctl`,
//!   which exercise the entire CTL surface through the
//!   `ctl_to_mu`-based adapter.
//! - The MCC benchmark suite (`tests/mcc_benchmarks.rs`) which is a
//!   no-regression floor for any change to the engine.
//!
//! ## CTL-to-mu translation (see [`ctl_to_mu`])
//!
//! - `EX p`        →  `◇p`
//! - `AX p`        →  `□p`
//! - `EF p`        →  `μZ. p ∨ ◇Z`
//! - `AF p`        →  `μZ. p ∨ (◇true ∧ □Z)`  (MCC max-path: deadlocks
//!   with `p` false fall outside the lfp)
//! - `EG p`        →  `νZ. p ∧ (◇Z ∨ ¬◇true)` (deadlock with `p` true
//!   stays in the gfp)
//! - `AG p`        →  `νZ. p ∧ □Z`            (□ at deadlock is True
//!   vacuously)
//! - `E[p U q]`    →  `μZ. q ∨ (p ∧ ◇Z)`
//! - `A[p U q]`    →  `μZ. q ∨ (p ∧ ◇true ∧ □Z)`  (MCC max-path)
//!
//! Correctness argument: each encoding matches the standard
//! Emerson-Clarke characterisation of CTL in mu-calculus, with the two
//! MCC-specific corrections (◇true conjunct on `AF` and `AU` to
//! exclude deadlocks where the eventually-target was never met; the
//! `¬◇true` disjunct on `EG` to keep deadlock states where `p` holds
//! inside the gfp). The two corrections are exactly the deadlock
//! treatment the full-graph `tla-mc-core::ctl::CtlEngine` implements
//! (`gfp_eg` keeps deadlocks with `sat` true; `AF` rewrites to `Not
//! (EG (Not _))` and inherits the deadlock treatment by duality), so
//! the differential test
//! [`mu_calculus_tests::test_unified_matches_full_graph_oracle`]
//! provides end-to-end soundness evidence.
//!
//! ## Path forward for LTL / Liveness / StableMarking
//!
//! - **LTL**: translate an LTL formula to a Büchi or alternating
//!   automaton, take the product with the state space, and encode
//!   acceptance as `νZ. accept ∧ μY. Z ∧ ◇(accept ∨ Y)`-style mu
//!   formulas. The solver handles this directly once the encoding is
//!   built.
//! - **Liveness** (some marking is enabled-and-firable infinitely
//!   often): `νZ. μY. fireable ∧ ◇Z ∨ ◇Y` per transition; conjunction
//!   over all transitions.
//! - **StableMarking** (some marking is reached and persists):
//!   `EF(AG(stable))` where `stable` is the place-vector equality
//!   atom; reuses the CTL-style translation directly.
//!
//! All four migrations are pure encoding work — no solver changes.

use std::collections::VecDeque;
use std::time::Instant;

use rustc_hash::FxHashMap;
use thiserror::Error;

use crate::explorer::fingerprint::fingerprint_marking;
use crate::explorer::{ExplorationConfig, ExplorationSetup};
use crate::marking::{pack_marking_config, unpack_marking_config, MarkingConfig};
use crate::petri_net::{PetriNet, TransitionIdx};
use crate::resolved_predicate::eval_predicate;

/// Default poll interval for the wall-clock deadline. Matches the
/// other local Petri-net engines so the solver behaves consistently
/// under tight budgets.
const DEADLINE_POLL_INTERVAL: u32 = 4096;

/// Default node cap when no explicit cap is configured. Sized so the
/// EDG never sustains more memory than the regular full-graph
/// explorer at its default `max_states`. This is the *upper* clamp;
/// the effective cap is derived from available memory (see
/// [`LocalMuSolver::memory_budgeted_node_cap`]).
const DEFAULT_NODE_CAP: usize = 1_000_000;

/// Lower clamp for the memory-budgeted node cap. Always allow at least
/// this many EDG nodes so tiny-memory misdetection cannot starve the
/// solver into an immediate abort on trivial nets.
const MIN_NODE_CAP: usize = 50_000;

/// Conservative estimate of the heap cost of ONE EDG node, independent
/// of the net's marking width:
///   - `EdgNode` itself (state, subformula, mark, expanded, dep_head)
///     plus amortised `dep_pool` chain links (8 B per dependency edge):
///     ~64 B (a deliberate over-estimate since the S7 intrusive-chain
///     arena removed the per-node `Vec` header and allocation).
///   - one `node_index` FxHashMap entry ((u32,u32)->u32 + slot/load-
///     factor overhead): ~48 B.
const EDG_NODE_FIXED_BYTES: usize = 64 + 48;

/// Per-distinct-state heap cost in the shared [`StateSpace`], excluding
/// the marking payload (added separately from the net's place count):
///   - `markings` Box<[u8]> header: 16 B.
///   - `state_ids` FxHashMap<u128,u32> entry: ~32 B.
///   - `succ_span` entry + amortised `succ_pool` payload: ~32 B (a
///     deliberate over-estimate since the S8 side arena removed the
///     per-state `Box<[u32]>` header and allocation).
const STATE_SPACE_FIXED_BYTES_PER_STATE: usize = 16 + 32 + 32;

/// Fraction of currently-available memory the solver is permitted to
/// budget for its EDG + state space. Deliberately small: the solver
/// runs inside a process that also holds reduced nets, BMC solvers,
/// and (in the liveness path) is invoked once per (colored) transition
/// group, so the steady-state footprint must stay well under the host
/// limit to avoid the allocator-abort that loses every other
/// examination's work.
const NODE_CAP_MEMORY_FRACTION: f64 = 0.10;

/// Identifier for a fixpoint-bound variable. Variables are introduced
/// by [`MuFormula::Mu`] / [`MuFormula::Nu`] and referenced by
/// [`MuFormula::Var`]. The encoding scheme is purely positional /
/// caller-driven: any two formulas with distinct binders should use
/// distinct `VarId`s; the solver does not perform alpha-renaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct VarId(pub(crate) u32);

/// Fixpoint polarity. Least-fixpoint variables default to `False`
/// when the worklist drains; greatest-fixpoint variables default to
/// `True`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Polarity {
    /// μ — least fixpoint, default `False`.
    Mu,
    /// ν — greatest fixpoint, default `True`.
    Nu,
}

/// A modal mu-calculus formula over a labelled transition system,
/// generic over the atomic-proposition payload `A`. The Petri-net
/// instantiation uses `A = ResolvedPredicate`.
///
/// The current AST uses unlabelled modalities (`Diamond` and `Box`
/// quantify over all outgoing transitions). Action-indexed
/// modalities can be added later without changing the solver's
/// hyperedge structure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MuFormula<A> {
    /// Atomic state predicate.
    Atom(A),
    /// Reference to a fixpoint-bound variable.
    Var(VarId),
    /// Boolean negation. The formula `Not(Var(_))` is rejected by the
    /// solver — negation must be pushed inside before binding any
    /// variable used negatively, because the alternation-free
    /// algorithm assumes positive variable occurrences.
    Not(Box<MuFormula<A>>),
    /// Conjunction of zero or more children. Empty And is `true`.
    And(Vec<MuFormula<A>>),
    /// Disjunction of zero or more children. Empty Or is `false`.
    Or(Vec<MuFormula<A>>),
    /// `◇φ`: some successor satisfies `φ`. At a deadlock state, the
    /// modality is `false` (no successor exists).
    Diamond(Box<MuFormula<A>>),
    /// `□φ`: all successors satisfy `φ`. At a deadlock state, the
    /// modality is `true` (vacuously universally quantified).
    Box(Box<MuFormula<A>>),
    /// `μX. φ`: least fixpoint binding `X` in `φ`.
    Mu(VarId, Box<MuFormula<A>>),
    /// `νX. φ`: greatest fixpoint binding `X` in `φ`.
    Nu(VarId, Box<MuFormula<A>>),
}

/// Abort modes for the unified solver. Every variant produces
/// `Verdict::CannotCompute` at the pipeline boundary.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MuAbort {
    /// EDG node count exceeded the configured cap.
    #[error("local mu-calculus solver exceeded node cap")]
    NodeCapReached,
    /// Underlying state interner hit its budget.
    #[error("local mu-calculus solver exceeded state budget")]
    StateLimitReached,
    /// Wall-clock deadline elapsed.
    #[error("local mu-calculus solver hit the deadline")]
    DeadlineExceeded,
    /// A formula contains `Not(Var(_))` (or any negated bound
    /// variable). The algorithm assumes positive variable occurrences;
    /// callers must push negations to the leaves of their encoding.
    #[error("local mu-calculus solver: negated fixpoint variable (non-positive normal form)")]
    NegatedVariable,
    /// A bound variable was referenced but not in scope. Indicates a
    /// translation bug at the caller.
    #[error("local mu-calculus solver: unbound variable {0:?}")]
    UnboundVariable(VarId),
    /// A μ/ν fixpoint with a strict alternation depth (≥ 2) was
    /// encountered. The current alternation-free closure cannot
    /// soundly resolve such nestings; future work will add a parity-
    /// game progress measure. The pipeline maps this to
    /// `Verdict::CannotCompute`.
    #[error("local mu-calculus solver: strict mu/nu alternation not yet supported")]
    UnsupportedAlternation,
    /// Firing a transition would overflow a place's `u64` token count (#22).
    /// The reachable state space contains a non-representable marking, so the
    /// solver declines (routed to `Verdict::CannotCompute`).
    #[error("local mu-calculus solver hit a token-count overflow")]
    TokenOverflow,
}

// ---------------------------------------------------------------------------
// Subformula table (structural canonicalisation)
// ---------------------------------------------------------------------------

/// Unique id of a *subformula occurrence* in the interned table.
///
/// Two structurally identical sub-ASTs do *not* automatically share
/// an id: the table uses pointer identity for fast lookup, matching
/// the behaviour of the original CTL EDG. Callers that want sharing
/// should canonicalise their formula before solving — sharing is a
/// performance optimisation, not a correctness one (the same
/// (state, subformula) pair always produces the same verdict).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SubformulaId(u32);

struct SubformulaTable<'f, A> {
    nodes: Vec<&'f MuFormula<A>>,
    by_ptr: FxHashMap<*const MuFormula<A>, SubformulaId>,
    /// Polarity (μ or ν) of the most recently-bound enclosing
    /// fixpoint that *uses* this subformula. For nodes that are
    /// themselves fixpoints, this is the polarity of the node itself.
    /// Used by the closure pass to pick the default mark for an
    /// `Unknown` node.
    polarity: Vec<Option<Polarity>>,
}

impl<'f, A> SubformulaTable<'f, A> {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            by_ptr: FxHashMap::default(),
            polarity: Vec::new(),
        }
    }

    fn intern(&mut self, formula: &'f MuFormula<A>, polarity: Option<Polarity>) -> SubformulaId {
        let key = std::ptr::from_ref(formula);
        if let Some(&id) = self.by_ptr.get(&key) {
            // First-binding polarity wins. Two-level interning by
            // (ptr, polarity) is not needed because in the
            // alternation-free fragment the enclosing polarity is
            // unique per AST node.
            return id;
        }
        let id = SubformulaId(self.nodes.len() as u32);
        self.nodes.push(formula);
        self.polarity.push(polarity);
        self.by_ptr.insert(key, id);
        id
    }

    fn get(&self, id: SubformulaId) -> &'f MuFormula<A> {
        self.nodes[id.0 as usize]
    }

    fn polarity_of(&self, id: SubformulaId) -> Option<Polarity> {
        self.polarity[id.0 as usize]
    }

    fn lookup_ptr(&self, formula: &MuFormula<A>) -> Option<SubformulaId> {
        self.by_ptr.get(&std::ptr::from_ref(formula)).copied()
    }
}

// ---------------------------------------------------------------------------
// Variable environment
// ---------------------------------------------------------------------------

/// Environment for fixpoint variables. Stores `(VarId, body)` pairs
/// in binding order so a `Var(x)` reference can resolve to the body
/// of the innermost binder for `x`.
///
/// Also records the polarity of each binder so the solver can
/// propagate the right default-mark choice (μ → False, ν → True) to
/// the body when interning it.
struct VarEnv<'f, A> {
    bindings: Vec<(VarId, &'f MuFormula<A>, Polarity)>,
}

impl<'f, A> VarEnv<'f, A> {
    fn new() -> Self {
        Self {
            bindings: Vec::new(),
        }
    }

    fn push(&mut self, var: VarId, body: &'f MuFormula<A>, polarity: Polarity) {
        self.bindings.push((var, body, polarity));
    }

    fn pop(&mut self) {
        let _ = self.bindings.pop();
    }

    fn lookup(&self, var: VarId) -> Option<(&'f MuFormula<A>, Polarity)> {
        self.bindings
            .iter()
            .rev()
            .find_map(|(v, b, p)| if *v == var { Some((*b, *p)) } else { None })
    }

    /// Returns true if any *enclosing* binder has the opposite
    /// polarity to `target`. Used to detect strict mu/nu alternation
    /// at binding time.
    fn has_opposite_enclosing(&self, target: Polarity) -> bool {
        let opp = match target {
            Polarity::Mu => Polarity::Nu,
            Polarity::Nu => Polarity::Mu,
        };
        self.bindings.iter().any(|(_, _, p)| *p == opp)
    }
}

// ---------------------------------------------------------------------------
// Three-valued mark + EDG node
// ---------------------------------------------------------------------------

/// Three-valued assignment used by the positive-only Liu-Smolka local
/// algorithm. `Unknown` is the lattice bottom; `True` and `False` are
/// both maximal and incomparable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mark {
    Unknown,
    True,
    False,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct EdgNodeId(u32);

/// Sentinel for "no dependency link" in the intrusive dependency chain
/// (audit S7): both an empty `dep_head` and the end of a chain.
const DEP_NONE: u32 = u32::MAX;

/// One link of a node's dependents chain, stored in the solver-level
/// `dep_pool` arena (audit S7) instead of a per-node `Vec<EdgNodeId>` —
/// collapsing per-node heap allocations into one pool and shrinking
/// `EdgNode` by the `Vec` header.
struct DepLink {
    dependent: EdgNodeId,
    /// Pool index of the next link, or [`DEP_NONE`].
    next: u32,
}

struct EdgNode {
    state: u32,
    subformula: SubformulaId,
    mark: Mark,
    expanded: bool,
    /// Head of this node's dependents chain in `dep_pool` (most recently
    /// added first), or [`DEP_NONE`] when empty / already consumed.
    dep_head: u32,
}

/// Detach node `node_id`'s dependents chain and collect it into `out`
/// in INSERTION order — exactly the order the former per-node
/// `Vec<EdgNodeId>` yielded under `std::mem::take` (the chain is stored
/// newest-first, so the walk is reversed). Wake order cannot change any
/// verdict (marks are write-once and the recheck rules are monotone),
/// but preserving it keeps exploration order — and thus cap/deadline
/// abort points — bit-identical to the pre-arena code.
fn take_deps(
    nodes: &mut [EdgNode],
    dep_pool: &[DepLink],
    node_id: EdgNodeId,
    out: &mut Vec<EdgNodeId>,
) {
    out.clear();
    let mut cur = std::mem::replace(&mut nodes[node_id.0 as usize].dep_head, DEP_NONE);
    while cur != DEP_NONE {
        let link = &dep_pool[cur as usize];
        out.push(link.dependent);
        cur = link.next;
    }
    out.reverse();
}

// ---------------------------------------------------------------------------
// Petri-net state space (shared with the legacy CTL EDG; the abort
// type is local so the unified solver does not depend on CTL-specific
// modules).
// ---------------------------------------------------------------------------

struct StateSpace<'a> {
    net: &'a PetriNet,
    max_states: usize,
    pack_capacity: usize,
    marking_config: MarkingConfig,
    state_ids: FxHashMap<u128, u32>,
    /// PACKED marking per state (the dedup key, ~1 byte/place for token-conserving
    /// nets vs 8 for a `Vec<u64>`) — decode via `marking_into`. Lossless, so state
    /// identity and every verdict are unchanged.
    markings: Vec<Box<[u8]>>,
    /// Cached sorted/deduped successor state ids, stored as one flat
    /// side arena (audit S8) instead of one `Option<Box<[u32]>>` per
    /// state: `succ_span[s]` is `(start, len)` into `succ_pool`, with
    /// `start == u32::MAX` meaning "not yet expanded".
    succ_pool: Vec<u32>,
    succ_span: Vec<(u32, u32)>,
    deadline: Option<Instant>,
    deadline_counter: u32,
}

/// `succ_span` sentinel: state not yet expanded.
const SUCC_UNEXPANDED: (u32, u32) = (u32::MAX, 0);

impl<'a> StateSpace<'a> {
    fn new(net: &'a PetriNet, config: &ExplorationConfig) -> Self {
        let setup = ExplorationSetup::analyze(net);
        let mut state_ids = FxHashMap::default();
        state_ids.insert(fingerprint_marking(&setup.initial_packed), 0);

        Self {
            net,
            max_states: config.max_states(),
            pack_capacity: setup.pack_capacity,
            marking_config: setup.marking_config,
            state_ids,
            markings: vec![setup.initial_packed],
            succ_pool: Vec::new(),
            succ_span: vec![SUCC_UNEXPANDED],
            deadline: config.deadline(),
            deadline_counter: 0,
        }
    }

    /// Repoint the deadline for a subsequent reuse of this (warm) state space
    /// across a new mu-formula, keeping every interned marking + cached successor
    /// span. The cache is purely the net's (formula-independent) transition
    /// relation, so reusing it is verdict-identical to a fresh build; only the
    /// wall budget for any *further* interning changes.
    fn set_deadline(&mut self, deadline: Option<Instant>) {
        self.deadline = deadline;
        self.deadline_counter = 0;
    }

    /// Decode state `state_id`'s marking into `out` (reused to avoid allocation).
    /// Markings are stored PACKED, so reads go through the (lossless) codec.
    fn marking_into(&self, state_id: u32, out: &mut Vec<u64>) {
        unpack_marking_config(&self.markings[state_id as usize], &self.marking_config, out);
    }

    fn intern_marking(&mut self, marking: &[u64], pack_buf: &mut Vec<u8>) -> Result<u32, MuAbort> {
        self.check_deadline()?;
        pack_marking_config(marking, &self.marking_config, pack_buf);
        let fingerprint = fingerprint_marking(pack_buf);
        if let Some(&existing) = self.state_ids.get(&fingerprint) {
            return Ok(existing);
        }
        if self.markings.len() >= self.max_states {
            return Err(MuAbort::StateLimitReached);
        }
        let state_id = self.markings.len() as u32;
        self.state_ids.insert(fingerprint, state_id);
        // Store the packed bytes already computed above for the fingerprint — not
        // the fat `Vec<u64>` (the ~8x EDG memory win).
        self.markings.push(pack_buf.as_slice().into());
        self.succ_span.push(SUCC_UNEXPANDED);
        Ok(state_id)
    }

    /// Successor slice of `state_id` if already expanded by
    /// [`Self::successors_of`], `None` otherwise — the former
    /// `successors[s].as_deref()`.
    fn successors_cached(&self, state_id: u32) -> Option<&[u32]> {
        let (start, len) = self.succ_span[state_id as usize];
        (start != u32::MAX).then(|| &self.succ_pool[start as usize..start as usize + len as usize])
    }

    fn successors_of(&mut self, state_id: u32) -> Result<&[u32], MuAbort> {
        if self.successors_cached(state_id).is_some() {
            return Ok(self.successors_cached(state_id).unwrap());
        }
        let mut current = Vec::new();
        unpack_marking_config(
            &self.markings[state_id as usize],
            &self.marking_config,
            &mut current,
        );
        let mut pack_buf = Vec::with_capacity(self.pack_capacity);
        let mut out: Vec<u32> = Vec::new();

        for tidx in 0..self.net.num_transitions() {
            let transition = TransitionIdx(tidx as u32);
            if !self.net.is_enabled(&current, transition) {
                continue;
            }
            // Fail-closed (#22): token-count overflow leaves `current` partially
            // mutated, so do NOT undo — decline the solver as inconclusive.
            self.net
                .apply_delta(&mut current, transition)
                .map_err(|_| MuAbort::TokenOverflow)?;
            let succ_id = self.intern_marking(&current, &mut pack_buf)?;
            self.net.undo_delta(&mut current, transition);
            out.push(succ_id);
        }

        out.sort_unstable();
        out.dedup();
        // Append to the side arena (audit S8). `intern_marking` above only
        // touches `succ_span`/`markings`, never `succ_pool`, so the pool is
        // stable across the exploration loop. The start offset must stay
        // strictly below the `u32::MAX` sentinel; fail closed (inconclusive)
        // rather than wrap, mirroring the state-count budget.
        let start = self.succ_pool.len();
        if u32::try_from(start + out.len()).is_err() || start as u64 >= u64::from(u32::MAX) {
            return Err(MuAbort::StateLimitReached);
        }
        self.succ_pool.extend_from_slice(&out);
        self.succ_span[state_id as usize] = (start as u32, out.len() as u32);
        Ok(&self.succ_pool[start..start + out.len()])
    }

    fn check_deadline(&mut self) -> Result<(), MuAbort> {
        self.deadline_counter = self.deadline_counter.wrapping_add(1);
        if self.deadline_counter.is_multiple_of(DEADLINE_POLL_INTERVAL)
            && self
                .deadline
                .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(MuAbort::DeadlineExceeded);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CTL → mu translation
// ---------------------------------------------------------------------------

/// Translate a [`super::ctl::resolve::ResolvedCtl`] into an
/// equivalent [`MuFormula`] using the standard Emerson-Clarke
/// encoding plus MCC maximal-path corrections at deadlocks.
///
/// See the module docstring for the full table of operator
/// equivalences. The returned formula has positive normal form (no
/// `Not` wraps a fixpoint variable) by construction — the encoding
/// only introduces `Not` around atoms (via the predicate AST, which
/// is unaffected) or around fixpoint formulas as a whole (which is
/// fine; only `Not` of a *bound variable* is rejected).
pub(crate) fn ctl_to_mu(
    formula: &super::ctl::resolve::ResolvedCtl,
) -> MuFormula<crate::resolved_predicate::ResolvedPredicate> {
    let mut counter: u32 = 0;
    ctl_to_mu_with_counter(formula, &mut counter)
}

fn fresh_var(counter: &mut u32) -> VarId {
    let v = VarId(*counter);
    *counter += 1;
    v
}

fn ctl_to_mu_with_counter(
    formula: &super::ctl::resolve::ResolvedCtl,
    counter: &mut u32,
) -> MuFormula<crate::resolved_predicate::ResolvedPredicate> {
    use super::ctl::resolve::ResolvedCtl as C;
    match formula {
        C::Atom(predicate) => MuFormula::Atom(predicate.clone()),
        C::Not(inner) => MuFormula::Not(Box::new(ctl_to_mu_with_counter(inner, counter))),
        C::And(children) => MuFormula::And(
            children
                .iter()
                .map(|c| ctl_to_mu_with_counter(c, counter))
                .collect(),
        ),
        C::Or(children) => MuFormula::Or(
            children
                .iter()
                .map(|c| ctl_to_mu_with_counter(c, counter))
                .collect(),
        ),
        C::EX(inner) => MuFormula::Diamond(Box::new(ctl_to_mu_with_counter(inner, counter))),
        C::AX(inner) => MuFormula::Box(Box::new(ctl_to_mu_with_counter(inner, counter))),
        C::EF(inner) => {
            // μZ. p ∨ ◇Z
            let z = fresh_var(counter);
            let p = ctl_to_mu_with_counter(inner, counter);
            MuFormula::Mu(
                z,
                Box::new(MuFormula::Or(vec![
                    p,
                    MuFormula::Diamond(Box::new(MuFormula::Var(z))),
                ])),
            )
        }
        C::AG(inner) => {
            // νZ. p ∧ □Z
            let z = fresh_var(counter);
            let p = ctl_to_mu_with_counter(inner, counter);
            MuFormula::Nu(
                z,
                Box::new(MuFormula::And(vec![
                    p,
                    MuFormula::Box(Box::new(MuFormula::Var(z))),
                ])),
            )
        }
        C::EG(inner) => {
            // νZ. p ∧ (◇Z ∨ ¬◇true)
            // The (¬◇true) disjunct preserves deadlock-with-p-true
            // states inside the gfp under MCC max-path semantics.
            let z = fresh_var(counter);
            let p = ctl_to_mu_with_counter(inner, counter);
            let has_succ = MuFormula::Diamond(Box::new(MuFormula::Atom(
                crate::resolved_predicate::ResolvedPredicate::True,
            )));
            MuFormula::Nu(
                z,
                Box::new(MuFormula::And(vec![
                    p,
                    MuFormula::Or(vec![
                        MuFormula::Diamond(Box::new(MuFormula::Var(z))),
                        MuFormula::Not(Box::new(has_succ)),
                    ]),
                ])),
            )
        }
        C::AF(inner) => {
            // μZ. p ∨ (◇true ∧ □Z)
            // The ◇true conjunct prevents deadlocks-without-p from
            // entering the lfp (MCC max-path: a maximal path that
            // ends in a deadlock without p does not satisfy AF p).
            let z = fresh_var(counter);
            let p = ctl_to_mu_with_counter(inner, counter);
            let has_succ = MuFormula::Diamond(Box::new(MuFormula::Atom(
                crate::resolved_predicate::ResolvedPredicate::True,
            )));
            MuFormula::Mu(
                z,
                Box::new(MuFormula::Or(vec![
                    p,
                    MuFormula::And(vec![has_succ, MuFormula::Box(Box::new(MuFormula::Var(z)))]),
                ])),
            )
        }
        C::EU(phi, psi) => {
            // μZ. q ∨ (p ∧ ◇Z)
            let z = fresh_var(counter);
            let p = ctl_to_mu_with_counter(phi, counter);
            let q = ctl_to_mu_with_counter(psi, counter);
            MuFormula::Mu(
                z,
                Box::new(MuFormula::Or(vec![
                    q,
                    MuFormula::And(vec![p, MuFormula::Diamond(Box::new(MuFormula::Var(z)))]),
                ])),
            )
        }
        C::AU(phi, psi) => {
            // μZ. q ∨ (p ∧ ◇true ∧ □Z)
            let z = fresh_var(counter);
            let p = ctl_to_mu_with_counter(phi, counter);
            let q = ctl_to_mu_with_counter(psi, counter);
            let has_succ = MuFormula::Diamond(Box::new(MuFormula::Atom(
                crate::resolved_predicate::ResolvedPredicate::True,
            )));
            MuFormula::Mu(
                z,
                Box::new(MuFormula::Or(vec![
                    q,
                    MuFormula::And(vec![
                        p,
                        has_succ,
                        MuFormula::Box(Box::new(MuFormula::Var(z))),
                    ]),
                ])),
            )
        }
        C::EGF(inner) => {
            // E(GF a): the Emerson–Lei fair cycle νZ. μY. (a ∧ ◇ˢZ) ∨ ◇ˢY with
            // the deadlock-stutter successor ◇ˢV = ◇V ∨ (¬◇true ∧ V), so a
            // deadlocked a-state is an infinite a-stutter witness — the same
            // deadlock convention the EG arm above uses (and the GPU/CPU
            // CtlEngine fair-cycle evaluators). Note this is a genuinely
            // ALTERNATING (νμ) formula.
            let z = fresh_var(counter);
            let y = fresh_var(counter);
            let p = ctl_to_mu_with_counter(inner, counter);
            // ◇ˢ Var = ◇Var ∨ (¬◇true ∧ Var).
            let ex_stutter = |var: VarId| {
                let deadlock = MuFormula::Not(Box::new(MuFormula::Diamond(Box::new(
                    MuFormula::Atom(crate::resolved_predicate::ResolvedPredicate::True),
                ))));
                MuFormula::Or(vec![
                    MuFormula::Diamond(Box::new(MuFormula::Var(var))),
                    MuFormula::And(vec![deadlock, MuFormula::Var(var)]),
                ])
            };
            MuFormula::Nu(
                z,
                Box::new(MuFormula::Mu(
                    y,
                    Box::new(MuFormula::Or(vec![
                        MuFormula::And(vec![p, ex_stutter(z)]),
                        ex_stutter(y),
                    ])),
                )),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Atom evaluator. The Petri-net instantiation uses
/// `PetriAtomEvaluator` defined below.
trait AtomEval<A> {
    fn eval(&mut self, state: u32, atom: &A) -> bool;
}

struct PetriAtomEvaluator<'a> {
    net: &'a PetriNet,
    state_space: &'a StateSpace<'a>,
    /// Reused buffer for decoding a packed marking during atom evaluation.
    scratch: Vec<u64>,
}

impl<'a> AtomEval<crate::resolved_predicate::ResolvedPredicate> for PetriAtomEvaluator<'a> {
    fn eval(&mut self, state: u32, atom: &crate::resolved_predicate::ResolvedPredicate) -> bool {
        self.state_space.marking_into(state, &mut self.scratch);
        eval_predicate(atom, &self.scratch, self.net)
    }
}

/// Liu-Smolka local mu-calculus solver instantiated for Petri nets.
///
/// Construct with [`Self::new`], then call [`Self::solve`] with the
/// formula to evaluate at the initial state. The solver consumes the
/// formula by reference and uses pointer identity for subformula
/// interning, so the caller must keep the formula alive until `solve`
/// returns.
pub(crate) struct LocalMuSolver<'a> {
    state_space: StateSpace<'a>,
    node_cap: usize,
    nodes: Vec<EdgNode>,
    node_index: FxHashMap<(u32, SubformulaId), EdgNodeId>,
    /// Dependency wake-ups: nodes whose child just received a final
    /// mark. Drained BEFORE `explore_work` so certain-True /
    /// certain-False back-propagation toward the root is immediate
    /// (the verifypn certain-zero "D before W" search strategy).
    dep_work: VecDeque<EdgNodeId>,
    /// Fresh-expansion frontier, popped LIFO (depth-first) so deep
    /// witnesses / refutations are reached without leveling the whole
    /// breadth frontier first. Queue order never affects the verdict:
    /// marks are write-once and every propagation rule is monotone in
    /// its children, so any drain order converges to the same
    /// assignment; order only affects how soon the root decides.
    explore_work: Vec<EdgNodeId>,
    /// Arena for the nodes' intrusive dependents chains (audit S7): one
    /// flat pool of links instead of one `Vec<EdgNodeId>` per node.
    dep_pool: Vec<DepLink>,
    /// Reused scratch for draining a node's dependents chain in
    /// [`take_deps`] without a per-`set_mark` allocation.
    wake_scratch: Vec<EdgNodeId>,
    /// Adaptive MEMORY probe (the wall-clock deadline is enforced separately by
    /// `StateSpace::check_deadline` on the marking-interning path). Ticked per
    /// EDG-node allocation in `ensure_node` — a pathological dependency/
    /// successor fan-out can outpace the per-node estimate by interning many
    /// uncompressed markings, so bytes, not node count, are the real bound.
    probe: tla_resource::MemoryProbe,
}

impl<'a> LocalMuSolver<'a> {
    pub(crate) fn new(net: &'a PetriNet, config: &ExplorationConfig) -> Self {
        Self {
            state_space: StateSpace::new(net, config),
            node_cap: Self::memory_budgeted_node_cap(net),
            nodes: Vec::new(),
            node_index: FxHashMap::default(),
            dep_work: VecDeque::new(),
            explore_work: Vec::new(),
            dep_pool: Vec::new(),
            wake_scratch: Vec::new(),
            probe: crate::memory::explorer_probe(None),
        }
    }

    /// Derive a node cap from currently-available memory so the EDG
    /// (plus its share of the [`StateSpace`]) cannot grow the process
    /// to an allocator abort. Returns [`DEFAULT_NODE_CAP`] when memory
    /// detection is unavailable, and always at least [`MIN_NODE_CAP`].
    ///
    /// The per-node estimate folds in the marking payload (one
    /// `Box<[u64]>` of `num_places` words per distinct state) because
    /// the state space is clamped to the node cap in [`Self::solve`],
    /// so each node accounts for at most one distinct state.
    pub(crate) fn memory_budgeted_node_cap(net: &PetriNet) -> usize {
        let Some(available) = crate::memory::available_memory_bytes() else {
            return DEFAULT_NODE_CAP;
        };
        let marking_bytes = net.num_places().saturating_mul(std::mem::size_of::<u64>());
        let bytes_per_node = EDG_NODE_FIXED_BYTES
            .saturating_add(STATE_SPACE_FIXED_BYTES_PER_STATE)
            .saturating_add(marking_bytes)
            .max(1);
        let budget = (available as f64 * NODE_CAP_MEMORY_FRACTION) as usize;
        (budget / bytes_per_node).clamp(MIN_NODE_CAP, DEFAULT_NODE_CAP)
    }

    pub(crate) fn with_node_cap(mut self, cap: usize) -> Self {
        self.node_cap = cap;
        self
    }

    /// Repoint the (per-formula) deadline for a reuse of this solver on a new
    /// mu-formula, keeping the warm shared [`StateSpace`]. See [`Self::solve`] and
    /// [`liveness_via_mu_calculus`]: solving each transition group with a fresh
    /// solver rebuilt the entire reachable state space per group (O(G·|R|) on
    /// nets with G≈|T| singleton groups, e.g. BridgeAndVehicles-PT ~970 groups);
    /// one reused solver builds the formula-independent successor cache ONCE and
    /// re-explores only each group's own EDG over it.
    pub(crate) fn set_deadline(&mut self, deadline: Option<Instant>) {
        self.state_space.set_deadline(deadline);
    }

    /// Clear the per-formula EDG so this solver can be reused for the next
    /// mu-formula over the SAME (warm) state space. Everything cleared here is a
    /// function of the formula only (nodes, the `(state, subformula)` index, both
    /// worklists, the dependents arena, and the memory probe); the state space
    /// itself — interned markings and cached successors — is deliberately kept.
    /// A stale EDG entry surviving would be a `(state_id, SubformulaId)` from the
    /// PRIOR formula whose pointer-identity `SubformulaId` cannot recur, so
    /// clearing is a correctness requirement, not just hygiene.
    fn reset_edg(&mut self) {
        self.nodes.clear();
        self.node_index.clear();
        self.dep_work.clear();
        self.explore_work.clear();
        self.dep_pool.clear();
        self.wake_scratch.clear();
        self.probe = crate::memory::explorer_probe(None);
    }

    /// Solve `formula` at the initial state.
    pub(crate) fn solve(
        &mut self,
        formula: &MuFormula<crate::resolved_predicate::ResolvedPredicate>,
    ) -> Result<bool, MuAbort> {
        // Reset the per-formula EDG so a REUSED solver (warm shared state space)
        // starts from a clean evaluation graph — a no-op on a freshly-constructed
        // solver, and the enabler of the one-state-space-per-net reuse in
        // `liveness_via_mu_calculus`.
        self.reset_edg();

        // Sanity: well-formedness (positive normal form for bound
        // variables, no strict alternation).
        Self::well_formed(formula, &mut VarEnv::new())?;

        // Clamp the shared state space to the EDG node budget so that
        // `successors_of` interning (which is bounded only by
        // `StateSpace::max_states`, NOT by `node_cap`) cannot grow the
        // process past the memory the node cap was sized for. A state
        // is interned only while expanding a Diamond/Box node, and the
        // expander immediately calls `ensure_node` for each successor;
        // once `node_cap` is reached `ensure_node` aborts, so the live
        // marking count exceeds `node_cap` by at most one successor
        // batch (<= num_transitions). Capping here therefore never
        // triggers a `StateLimitReached` that the existing `node_cap`
        // would not already raise as `NodeCapReached`; it only makes
        // the bound memory-proportional and fail-closed.
        let state_cap = self
            .node_cap
            .saturating_add(self.state_space.net.num_transitions())
            .max(1);
        if self.state_space.max_states > state_cap {
            self.state_space.max_states = state_cap;
        }

        let mut subformulas: SubformulaTable<'_, crate::resolved_predicate::ResolvedPredicate> =
            SubformulaTable::new();
        let mut var_env: VarEnv<'_, crate::resolved_predicate::ResolvedPredicate> = VarEnv::new();

        let root_sub = self.intern_recursive(formula, None, &mut subformulas, &mut var_env);
        let root = self.ensure_node(0, root_sub)?;
        self.explore_work.push(root);

        // Worklist phase. Terminates the instant the ROOT is decided
        // EITHER way (certain-True or certain-False): marks are
        // write-once and every derivation that sets one is sound on
        // its own (True marks come from witness derivations, False
        // marks from well-founded refutations bottoming at decided
        // leaves), so a decided root can never be revised by further
        // exploration.
        while let Some(node_id) = self.next_work() {
            self.expand_or_propagate(node_id, &mut subformulas, &mut var_env)?;
            if self.nodes[root.0 as usize].mark != Mark::Unknown {
                break;
            }
        }

        if self.nodes[root.0 as usize].mark == Mark::Unknown {
            self.fixpoint_close(root, &subformulas, &var_env);
        }

        match self.nodes[root.0 as usize].mark {
            Mark::True => Ok(true),
            Mark::False => Ok(false),
            // Should not happen: the close pass decides every fixpoint
            // node, and non-fixpoint nodes are decided during
            // expansion. Be defensive: report as unsupported so the
            // caller's pipeline can fall back.
            Mark::Unknown => Err(MuAbort::UnsupportedAlternation),
        }
    }

    /// Recursively intern every subformula. Each interned node carries
    /// the polarity of the most recently bound enclosing fixpoint (or
    /// `None` if there is no enclosing fixpoint).
    fn intern_recursive<'f>(
        &mut self,
        formula: &'f MuFormula<crate::resolved_predicate::ResolvedPredicate>,
        enclosing_polarity: Option<Polarity>,
        subformulas: &mut SubformulaTable<'f, crate::resolved_predicate::ResolvedPredicate>,
        _var_env: &mut VarEnv<'f, crate::resolved_predicate::ResolvedPredicate>,
    ) -> SubformulaId {
        let id = subformulas.intern(formula, enclosing_polarity);
        // The polarity for *child* nodes is the polarity of the
        // nearest enclosing fixpoint. When this node IS a fixpoint,
        // its body inherits the *new* polarity.
        let child_polarity = match formula {
            MuFormula::Mu(_, _) => Some(Polarity::Mu),
            MuFormula::Nu(_, _) => Some(Polarity::Nu),
            _ => enclosing_polarity,
        };
        match formula {
            MuFormula::Atom(_) | MuFormula::Var(_) => {}
            MuFormula::Not(inner) | MuFormula::Diamond(inner) | MuFormula::Box(inner) => {
                let _ = self.intern_recursive(inner, child_polarity, subformulas, _var_env);
            }
            MuFormula::And(children) | MuFormula::Or(children) => {
                for child in children {
                    let _ = self.intern_recursive(child, child_polarity, subformulas, _var_env);
                }
            }
            MuFormula::Mu(_, body) | MuFormula::Nu(_, body) => {
                let _ = self.intern_recursive(body, child_polarity, subformulas, _var_env);
            }
        }
        id
    }

    /// Well-formedness check: bound variables must appear positively
    /// (never under `Not`), and *strict* mu/nu alternation is
    /// rejected. Free variables are flagged as
    /// [`MuAbort::UnboundVariable`].
    ///
    /// Strict alternation here means: inside a fixpoint of polarity
    /// `P`, a `Var(y)` reference to an outer fixpoint of polarity
    /// `¬P` is forbidden. This is the standard alternation-free
    /// fragment AFMC. Pure nesting like
    /// `νZ. (μY. p ∨ ◇Y) ∧ □Z` is *fine* — Y is referenced only
    /// inside its own μ-scope; the outer Z is never referenced
    /// inside the inner μ. Strict alternation is e.g.
    /// `νZ. μY. ◇Z ∨ ◇Y` where Z is referenced from inside the μ.
    ///
    /// Implementation uses a lightweight `(VarId, Polarity)` stack
    /// instead of the full `VarEnv` so the borrow on the formula does
    /// not need to outlive a separate lifetime parameter.
    fn well_formed<A>(formula: &MuFormula<A>, _env: &mut VarEnv<'_, A>) -> Result<(), MuAbort> {
        let mut stack: Vec<(VarId, Polarity)> = Vec::new();
        Self::well_formed_inner(formula, &mut stack, true, None)
    }

    /// `enclosing_polarity` is `Some(p)` iff the recursion is strictly
    /// inside a fixpoint binder of polarity `p`. When we see a
    /// `Var(y)`, we check that y's binder polarity matches
    /// `enclosing_polarity`; a mismatch is strict alternation.
    fn well_formed_inner<A>(
        formula: &MuFormula<A>,
        stack: &mut Vec<(VarId, Polarity)>,
        positive: bool,
        enclosing_polarity: Option<Polarity>,
    ) -> Result<(), MuAbort> {
        match formula {
            MuFormula::Atom(_) => Ok(()),
            MuFormula::Var(v) => {
                if !positive {
                    return Err(MuAbort::NegatedVariable);
                }
                let var_polarity = stack.iter().find(|(b, _)| b == v).map(|(_, p)| *p);
                match var_polarity {
                    None => Err(MuAbort::UnboundVariable(*v)),
                    Some(var_p) => {
                        if let Some(enc_p) = enclosing_polarity {
                            if enc_p != var_p {
                                return Err(MuAbort::UnsupportedAlternation);
                            }
                        }
                        // Staged-closure discipline: a `Var` may reference only
                        // the *innermost* enclosing binder. The closure defaults
                        // fixpoint components innermost-first (descending
                        // binder-body id); a same-polarity reference to an
                        // *outer* sibling binder can invert that dependency
                        // order and freeze a wrong mark. Every `ctl_to_mu`
                        // formula satisfies this (each Var occurs solely in its
                        // own binder's immediate schema), so production never
                        // hits this abort — it statically excludes hand-built
                        // formulas the staged closure was not designed for.
                        if stack.last().map(|(b, _)| b) != Some(v) {
                            return Err(MuAbort::UnsupportedAlternation);
                        }
                        Ok(())
                    }
                }
            }
            MuFormula::Not(inner) => {
                Self::well_formed_inner(inner, stack, !positive, enclosing_polarity)
            }
            MuFormula::And(children) | MuFormula::Or(children) => {
                for c in children {
                    Self::well_formed_inner(c, stack, positive, enclosing_polarity)?;
                }
                Ok(())
            }
            MuFormula::Diamond(inner) | MuFormula::Box(inner) => {
                Self::well_formed_inner(inner, stack, positive, enclosing_polarity)
            }
            MuFormula::Mu(v, body) => {
                // Positivity of the bound variable is checked against
                // the body alone, not the outer context. A `Not` *outside*
                // the binder just complements the binder's verdict.
                stack.push((*v, Polarity::Mu));
                let result = Self::well_formed_inner(body, stack, true, Some(Polarity::Mu));
                stack.pop();
                let _ = positive;
                result
            }
            MuFormula::Nu(v, body) => {
                stack.push((*v, Polarity::Nu));
                let result = Self::well_formed_inner(body, stack, true, Some(Polarity::Nu));
                stack.pop();
                let _ = positive;
                result
            }
        }
    }

    fn ensure_node(&mut self, state: u32, subformula: SubformulaId) -> Result<EdgNodeId, MuAbort> {
        if let Some(&id) = self.node_index.get(&(state, subformula)) {
            return Ok(id);
        }
        if self.nodes.len() >= self.node_cap {
            return Err(MuAbort::NodeCapReached);
        }
        // Live memory guard: even within the node-COUNT budget, a pathological
        // dependency/successor fan-out can outpace the per-node estimate by
        // interning many uncompressed markings. The adaptive probe fails closed
        // (CannotCompute) long before the allocator would abort the process.
        if self.probe.over_budget() {
            return Err(MuAbort::NodeCapReached);
        }
        let id = EdgNodeId(self.nodes.len() as u32);
        self.nodes.push(EdgNode {
            state,
            subformula,
            mark: Mark::Unknown,
            expanded: false,
            dep_head: DEP_NONE,
        });
        self.node_index.insert((state, subformula), id);
        Ok(id)
    }

    fn lookup_node(&self, state: u32, subformula: SubformulaId) -> Option<EdgNodeId> {
        self.node_index.get(&(state, subformula)).copied()
    }

    fn add_dependency(&mut self, target: EdgNodeId, dependent: EdgNodeId) {
        // Chain-walk dedup — the same wake-count-only guard the former
        // `Vec::contains` provided (duplicate wakes are harmless to the
        // verdict; marks are write-once).
        let head = self.nodes[target.0 as usize].dep_head;
        let mut cur = head;
        while cur != DEP_NONE {
            let link = &self.dep_pool[cur as usize];
            if link.dependent == dependent {
                return;
            }
            cur = link.next;
        }
        let idx = u32::try_from(self.dep_pool.len()).expect("EDG dependency pool overflow");
        assert!(idx != DEP_NONE, "EDG dependency pool overflow");
        self.dep_pool.push(DepLink {
            dependent,
            next: head,
        });
        self.nodes[target.0 as usize].dep_head = idx;
    }

    /// Pop the next worklist entry: dependency wake-ups first (so a
    /// freshly decided child immediately re-checks its parents, both
    /// for certain-True and certain-False), then the newest frontier
    /// node (DFS).
    fn next_work(&mut self) -> Option<EdgNodeId> {
        self.dep_work
            .pop_front()
            .or_else(|| self.explore_work.pop())
    }

    fn set_mark(&mut self, node_id: EdgNodeId, mark: Mark) {
        debug_assert!(mark != Mark::Unknown);
        if self.nodes[node_id.0 as usize].mark != Mark::Unknown {
            return;
        }
        self.nodes[node_id.0 as usize].mark = mark;
        let mut woken = std::mem::take(&mut self.wake_scratch);
        take_deps(&mut self.nodes, &self.dep_pool, node_id, &mut woken);
        for &dep in &woken {
            if self.nodes[dep.0 as usize].mark == Mark::Unknown {
                self.dep_work.push_back(dep);
            }
        }
        self.wake_scratch = woken;
    }

    fn expand_or_propagate<'f>(
        &mut self,
        node_id: EdgNodeId,
        subformulas: &mut SubformulaTable<'f, crate::resolved_predicate::ResolvedPredicate>,
        var_env: &mut VarEnv<'f, crate::resolved_predicate::ResolvedPredicate>,
    ) -> Result<(), MuAbort> {
        if self.nodes[node_id.0 as usize].mark != Mark::Unknown {
            return Ok(());
        }
        if !self.nodes[node_id.0 as usize].expanded {
            self.nodes[node_id.0 as usize].expanded = true;
            self.expand_node(node_id, subformulas, var_env)?;
        } else {
            self.recheck_node(node_id, subformulas, var_env)?;
        }
        Ok(())
    }

    /// First-time expansion: install hyperedges and try an immediate
    /// recheck for any node already-decided leaves.
    fn expand_node<'f>(
        &mut self,
        node_id: EdgNodeId,
        subformulas: &mut SubformulaTable<'f, crate::resolved_predicate::ResolvedPredicate>,
        var_env: &mut VarEnv<'f, crate::resolved_predicate::ResolvedPredicate>,
    ) -> Result<(), MuAbort> {
        let state = self.nodes[node_id.0 as usize].state;
        let sub_id = self.nodes[node_id.0 as usize].subformula;
        let formula = subformulas.get(sub_id);

        match formula {
            MuFormula::Atom(predicate) => {
                let mut evaluator = PetriAtomEvaluator {
                    net: self.state_space.net,
                    state_space: &self.state_space,
                    scratch: Vec::new(),
                };
                let value = evaluator.eval(state, predicate);
                self.set_mark(node_id, if value { Mark::True } else { Mark::False });
            }
            MuFormula::Var(v) => {
                // A bare Var node redirects to the body of the binder
                // via a dependency. The body's node (same state, same
                // sub_id for body) carries the real evaluation.
                let (body, _polarity) = var_env.lookup(*v).ok_or(MuAbort::UnboundVariable(*v))?;
                let body_id = subformulas
                    .lookup_ptr(body)
                    .expect("Var: body was interned during fixpoint setup");
                let dep = self.ensure_node(state, body_id)?;
                self.add_dependency(dep, node_id);
                self.explore_work.push(dep);
                self.recheck_node(node_id, subformulas, var_env)?;
            }
            MuFormula::Not(inner) => {
                let inner_id = subformulas
                    .lookup_ptr(inner)
                    .expect("Not child was interned");
                let dep = self.ensure_node(state, inner_id)?;
                self.add_dependency(dep, node_id);
                self.explore_work.push(dep);
                self.recheck_node(node_id, subformulas, var_env)?;
            }
            MuFormula::And(children) | MuFormula::Or(children) => {
                let mut dep_ids = Vec::with_capacity(children.len());
                for child in children {
                    let child_sub = subformulas
                        .lookup_ptr(child)
                        .expect("And/Or child was interned");
                    let dep = self.ensure_node(state, child_sub)?;
                    self.add_dependency(dep, node_id);
                    dep_ids.push(dep);
                }
                for dep in dep_ids {
                    self.explore_work.push(dep);
                }
                self.recheck_node(node_id, subformulas, var_env)?;
            }
            MuFormula::Diamond(inner) | MuFormula::Box(inner) => {
                let inner_id = subformulas
                    .lookup_ptr(inner)
                    .expect("Diamond/Box child was interned");
                let succs: Vec<u32> = self.state_space.successors_of(state)?.to_vec();
                if succs.is_empty() {
                    // Modal at a deadlock:
                    //   Diamond: False (no successor exists).
                    //   Box: True (vacuously universally quantified).
                    let mark = match formula {
                        MuFormula::Diamond(_) => Mark::False,
                        MuFormula::Box(_) => Mark::True,
                        _ => unreachable!(),
                    };
                    self.set_mark(node_id, mark);
                    return Ok(());
                }
                let mut dep_ids = Vec::with_capacity(succs.len());
                for s in succs {
                    let dep = self.ensure_node(s, inner_id)?;
                    self.add_dependency(dep, node_id);
                    dep_ids.push(dep);
                }
                for dep in dep_ids {
                    self.explore_work.push(dep);
                }
                self.recheck_node(node_id, subformulas, var_env)?;
            }
            MuFormula::Mu(v, body) | MuFormula::Nu(v, body) => {
                let polarity = match formula {
                    MuFormula::Mu(_, _) => Polarity::Mu,
                    MuFormula::Nu(_, _) => Polarity::Nu,
                    _ => unreachable!(),
                };
                let body_id = subformulas
                    .lookup_ptr(body)
                    .expect("fixpoint body was interned");
                // Push the variable binding so any Var(v) below
                // resolves correctly.
                var_env.push(*v, body, polarity);
                let dep = self.ensure_node(state, body_id)?;
                self.add_dependency(dep, node_id);
                self.explore_work.push(dep);
                // Note: we deliberately do not pop the binding here.
                // It remains in scope for the lifetime of the solve
                // call, because the EDG resolves dependent (state,
                // subformula) pairs lazily; popping would break
                // subsequent Var lookups for the same binder
                // appearing at other states.
                self.recheck_node(node_id, subformulas, var_env)?;
            }
        }
        Ok(())
    }

    /// Re-check a node's hyperedges with current dependent marks.
    fn recheck_node<'f>(
        &mut self,
        node_id: EdgNodeId,
        subformulas: &mut SubformulaTable<'f, crate::resolved_predicate::ResolvedPredicate>,
        var_env: &mut VarEnv<'f, crate::resolved_predicate::ResolvedPredicate>,
    ) -> Result<(), MuAbort> {
        if self.nodes[node_id.0 as usize].mark != Mark::Unknown {
            return Ok(());
        }
        let state = self.nodes[node_id.0 as usize].state;
        let sub_id = self.nodes[node_id.0 as usize].subformula;
        let formula = subformulas.get(sub_id);

        match formula {
            MuFormula::Atom(_) => {
                // Resolved during expand_node.
            }
            MuFormula::Var(v) => {
                let (body, _polarity) = var_env.lookup(*v).ok_or(MuAbort::UnboundVariable(*v))?;
                let body_id = subformulas
                    .lookup_ptr(body)
                    .expect("Var: body was interned during fixpoint setup");
                if let Some(dep) = self.lookup_node(state, body_id) {
                    match self.nodes[dep.0 as usize].mark {
                        Mark::True => self.set_mark(node_id, Mark::True),
                        Mark::False => self.set_mark(node_id, Mark::False),
                        Mark::Unknown => {}
                    }
                }
            }
            MuFormula::Not(inner) => {
                let inner_id = subformulas
                    .lookup_ptr(inner)
                    .expect("Not child was interned");
                if let Some(dep) = self.lookup_node(state, inner_id) {
                    match self.nodes[dep.0 as usize].mark {
                        Mark::True => self.set_mark(node_id, Mark::False),
                        Mark::False => self.set_mark(node_id, Mark::True),
                        Mark::Unknown => {}
                    }
                }
            }
            MuFormula::And(children) => {
                let mut all_true = true;
                for child in children {
                    let child_sub = subformulas
                        .lookup_ptr(child)
                        .expect("And child was interned");
                    let dep = self.lookup_node(state, child_sub);
                    match dep.map(|d| self.nodes[d.0 as usize].mark) {
                        Some(Mark::False) => {
                            self.set_mark(node_id, Mark::False);
                            return Ok(());
                        }
                        Some(Mark::True) => {}
                        Some(Mark::Unknown) | None => all_true = false,
                    }
                }
                if all_true {
                    self.set_mark(node_id, Mark::True);
                }
            }
            MuFormula::Or(children) => {
                let mut all_false = true;
                for child in children {
                    let child_sub = subformulas
                        .lookup_ptr(child)
                        .expect("Or child was interned");
                    let dep = self.lookup_node(state, child_sub);
                    match dep.map(|d| self.nodes[d.0 as usize].mark) {
                        Some(Mark::True) => {
                            self.set_mark(node_id, Mark::True);
                            return Ok(());
                        }
                        Some(Mark::False) => {}
                        Some(Mark::Unknown) | None => all_false = false,
                    }
                }
                if all_false {
                    self.set_mark(node_id, Mark::False);
                }
            }
            MuFormula::Diamond(inner) => {
                let inner_id = subformulas
                    .lookup_ptr(inner)
                    .expect("Diamond child was interned");
                let succs = self
                    .state_space
                    .successors_cached(state)
                    .expect("Diamond expands successors during expand_node");
                let mut all_false = true;
                for &s in succs {
                    let dep = self.lookup_node(s, inner_id);
                    match dep.map(|d| self.nodes[d.0 as usize].mark) {
                        Some(Mark::True) => {
                            self.set_mark(node_id, Mark::True);
                            return Ok(());
                        }
                        Some(Mark::False) => {}
                        Some(Mark::Unknown) | None => all_false = false,
                    }
                }
                if all_false {
                    self.set_mark(node_id, Mark::False);
                }
            }
            MuFormula::Box(inner) => {
                let inner_id = subformulas
                    .lookup_ptr(inner)
                    .expect("Box child was interned");
                let succs = self
                    .state_space
                    .successors_cached(state)
                    .expect("Box expands successors during expand_node");
                let mut all_true = true;
                for &s in succs {
                    let dep = self.lookup_node(s, inner_id);
                    match dep.map(|d| self.nodes[d.0 as usize].mark) {
                        Some(Mark::False) => {
                            self.set_mark(node_id, Mark::False);
                            return Ok(());
                        }
                        Some(Mark::True) => {}
                        Some(Mark::Unknown) | None => all_true = false,
                    }
                }
                if all_true {
                    self.set_mark(node_id, Mark::True);
                }
            }
            MuFormula::Mu(_, body) | MuFormula::Nu(_, body) => {
                // A fixpoint node inherits the mark of its body's
                // evaluation at the same state. The body's node is
                // installed during expand_node; here we read it.
                let body_id = subformulas
                    .lookup_ptr(body)
                    .expect("fixpoint body was interned");
                if let Some(dep) = self.lookup_node(state, body_id) {
                    match self.nodes[dep.0 as usize].mark {
                        Mark::True => self.set_mark(node_id, Mark::True),
                        Mark::False => self.set_mark(node_id, Mark::False),
                        Mark::Unknown => {}
                    }
                }
            }
        }
        Ok(())
    }

    /// Close any still-`Unknown` node by applying its polarity
    /// default and propagating.
    ///
    /// Strategy, designed to handle nested-but-non-strict fixpoints
    /// (e.g. `νZ. (μY. ...) ∧ □Z` from `AG(EF p)`) correctly:
    ///
    /// 1. Default `Var` nodes to their binder's polarity,
    ///    **one fixpoint component at a time, innermost binder
    ///    first**, propagating to stability between components. The
    ///    `Var` nodes are the cycle-breakers: a `Var(y)` resolves the
    ///    body-of-binder-y self-reference. Other fixpoint nodes
    ///    (`Mu(_, body)` / `Nu(_, body)`) are *not* defaulted —
    ///    they inherit from the body via propagation.
    /// 2. Once every component is settled, default any
    ///    *still*-Unknown `Mu`/`Nu` nodes (innermost-first) and
    ///    propagate again, until no fixpoint node remains Unknown.
    ///
    /// ## Why component staging is required for soundness
    ///
    /// At drain time, the worklist phase has derived every mark with
    /// a finite (well-founded) derivation. For a μ component this
    /// makes the polarity default *exact*: a still-Unknown μ-Var has
    /// no witness, hence is outside the least fixpoint — but ONLY
    /// once the values its component reads from *inner* components
    /// are final. Defaulting all components simultaneously (the
    /// previous behaviour) is unsound for nested fixpoints of mixed
    /// polarity: in `νZ. (μY. p ∨ ◇Y) ∧ □Z`, an outer `Var(Z)`
    /// defaulted True before the inner μ refutation (`EF p` False at
    /// some reachable state) has propagated will freeze the gfp at
    /// True — a wrong verdict (marks are write-once). Staging
    /// innermost-first lets each component consume the *final* values
    /// of its inner components, which is the standard bottom-up
    /// component evaluation of alternation-free mu-calculus
    /// (Cleaveland–Steffen; Dalsgaard et al.'s certain-zero treats
    /// the inner component's CZERO exactly this way. The inner-ν dual
    /// — e.g. `EF(AG p)` — works symmetrically: inner ν-Vars default
    /// True first, the derived `AG p` True states then *witness* the
    /// outer μ through Var-inherits-body propagation below.)
    ///
    /// Components are independent unless nested (a `Var` crossing
    /// into an *enclosing* binder's component requires equal polarity
    /// in the alternation-free fragment, and equal-polarity defaults
    /// commute), so ordering siblings arbitrarily is sound; ordering
    /// nested chains innermost-first is exactly the component
    /// dependency order.
    ///
    /// ## Propagation strategy + early exit (certain-zero closure)
    ///
    /// Propagation is *dependency-driven*: still-`Unknown` nodes have
    /// never had their dependents chains consumed (`set_mark` only fires
    /// on a final mark), so every node that could become decidable
    /// when `n` decides is registered in `n`'s chain. Seeding a worklist
    /// from the dependents of each newly defaulted/decided node and
    /// re-checking only those is therefore complete, and costs
    /// O(edges) instead of the previous O(nodes × EDG-depth) repeated
    /// full-array sweeps.
    ///
    /// Recheck order within a drain cannot change the outcome: marks
    /// are write-once, and each recheck rule is *consistent* — if a
    /// node is decidable to value v at child-marking m, it remains
    /// decidable to the same v at any refinement of m (an Or decided
    /// True keeps its True child; an Or decided False has all
    /// children False and they stay False; dually for And/Box/
    /// Diamond). Hence each component-drain fixpoint is unique.
    ///
    /// The closure terminates the moment the ROOT node is decided
    /// either way: a decided mark can never be revised, so the
    /// remaining propagation cannot change the verdict.
    fn fixpoint_close<'f>(
        &mut self,
        root: EdgNodeId,
        subformulas: &SubformulaTable<'f, crate::resolved_predicate::ResolvedPredicate>,
        var_env: &VarEnv<'f, crate::resolved_predicate::ResolvedPredicate>,
    ) {
        let mut work: VecDeque<EdgNodeId> = VecDeque::new();

        // Pass 1: collect still-Unknown Var nodes, keyed by their
        // binder's component (proxied by the binder *body*'s
        // subformula id: pre-order interning makes nested-inner
        // bodies strictly larger than their enclosing binder's, so
        // descending order is innermost-first along nesting chains;
        // sibling components are unordered and independent).
        let mut var_defaults: Vec<(u32, EdgNodeId, Mark)> = Vec::new();
        for node_id in 0..self.nodes.len() {
            if self.nodes[node_id].mark != Mark::Unknown {
                continue;
            }
            let sub_id = self.nodes[node_id].subformula;
            let MuFormula::Var(v) = subformulas.get(sub_id) else {
                continue;
            };
            let polarity = subformulas.polarity_of(sub_id).unwrap_or(Polarity::Mu);
            let mark = match polarity {
                Polarity::Mu => Mark::False,
                Polarity::Nu => Mark::True,
            };
            let component = var_env
                .lookup(*v)
                .and_then(|(body, _)| subformulas.lookup_ptr(body))
                .map_or(0, |body_id| body_id.0);
            var_defaults.push((component, EdgNodeId(node_id as u32), mark));
        }
        var_defaults.sort_by(|a, b| b.0.cmp(&a.0).then(a.1 .0.cmp(&b.1 .0)));

        // Default one component at a time (innermost first),
        // propagating to a fixed point in between so outer components
        // observe final inner values. Note a Var may have been
        // decided by an earlier component's propagation (Var inherits
        // its body's mark in `closure_recheck`); `close_mark` is a
        // no-op then.
        let mut idx = 0;
        while idx < var_defaults.len() {
            let component = var_defaults[idx].0;
            while idx < var_defaults.len() && var_defaults[idx].0 == component {
                let (_, node_id, mark) = var_defaults[idx];
                self.close_mark(node_id, mark, &mut work);
                idx += 1;
            }
            if self.drain_closure(root, &mut work, subformulas, var_env) {
                return;
            }
        }

        // Pass 2: default any still-Unknown fixpoint nodes
        // (innermost-first) and re-propagate.
        //
        // The closure allocates no new nodes, so the candidate set is
        // fixed; collecting it once and walking it in (subformula-id
        // descending, node-id ascending) order — skipping entries a
        // previous drain already decided — selects exactly the same
        // node at each step as the historical full-array rescan
        // (which picked the first node id among those with the
        // maximal subformula id). Subformula ids were allocated in
        // pre-order during `intern_recursive`, so larger ids
        // correspond to deeper subformulas. Defaulting deeper nodes
        // first lets the outer fixpoints inherit the right value via
        // propagation rather than picking up the polarity default
        // themselves.
        let mut candidates: Vec<(u32, EdgNodeId, Polarity)> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.mark == Mark::Unknown)
            .filter_map(|(node_id, node)| {
                let polarity = match subformulas.get(node.subformula) {
                    MuFormula::Mu(_, _) => Polarity::Mu,
                    MuFormula::Nu(_, _) => Polarity::Nu,
                    _ => return None,
                };
                Some((node.subformula.0, EdgNodeId(node_id as u32), polarity))
            })
            .collect();
        candidates.sort_by(|a, b| b.0.cmp(&a.0).then(a.1 .0.cmp(&b.1 .0)));

        for (_, node_id, polarity) in candidates {
            if self.nodes[node_id.0 as usize].mark != Mark::Unknown {
                continue;
            }
            let mark = match polarity {
                Polarity::Mu => Mark::False,
                Polarity::Nu => Mark::True,
            };
            self.close_mark(node_id, mark, &mut work);
            if self.drain_closure(root, &mut work, subformulas, var_env) {
                return;
            }
        }
    }

    /// Closure-phase mark write: like [`Self::set_mark`] but routes
    /// woken dependents into the closure worklist instead of the
    /// solver's global queues.
    fn close_mark(&mut self, node_id: EdgNodeId, mark: Mark, work: &mut VecDeque<EdgNodeId>) {
        debug_assert!(mark != Mark::Unknown);
        if self.nodes[node_id.0 as usize].mark != Mark::Unknown {
            return;
        }
        self.nodes[node_id.0 as usize].mark = mark;
        let mut woken = std::mem::take(&mut self.wake_scratch);
        take_deps(&mut self.nodes, &self.dep_pool, node_id, &mut woken);
        for &dep in &woken {
            if self.nodes[dep.0 as usize].mark == Mark::Unknown {
                work.push_back(dep);
            }
        }
        self.wake_scratch = woken;
    }

    /// Drain the closure worklist: re-check each woken node and, when
    /// it becomes decidable, mark it and wake its dependents in turn.
    /// Returns `true` (stop everything) the moment the root is
    /// decided — marks are final, so the verdict cannot change.
    fn drain_closure<'f>(
        &mut self,
        root: EdgNodeId,
        work: &mut VecDeque<EdgNodeId>,
        subformulas: &SubformulaTable<'f, crate::resolved_predicate::ResolvedPredicate>,
        var_env: &VarEnv<'f, crate::resolved_predicate::ResolvedPredicate>,
    ) -> bool {
        while let Some(node_id) = work.pop_front() {
            if self.nodes[root.0 as usize].mark != Mark::Unknown {
                return true;
            }
            if self.nodes[node_id.0 as usize].mark != Mark::Unknown {
                continue;
            }
            if let Some(mark) = self.closure_recheck(node_id, subformulas, var_env) {
                self.close_mark(node_id, mark, work);
            }
        }
        self.nodes[root.0 as usize].mark != Mark::Unknown
    }

    /// Side-effect-free recheck used by the close pass: compute the
    /// mark `node_id` is entitled to under the current child marks,
    /// or `None` when still undetermined. Children that were never
    /// materialised (closure does not expand) count as `Unknown`,
    /// exactly as in the worklist-phase rules.
    fn closure_recheck<'f>(
        &self,
        node_id: EdgNodeId,
        subformulas: &SubformulaTable<'f, crate::resolved_predicate::ResolvedPredicate>,
        var_env: &VarEnv<'f, crate::resolved_predicate::ResolvedPredicate>,
    ) -> Option<Mark> {
        let state = self.nodes[node_id.0 as usize].state;
        let sub_id = self.nodes[node_id.0 as usize].subformula;
        let formula = subformulas.get(sub_id);

        let child_mark =
            |state: u32, child: &MuFormula<crate::resolved_predicate::ResolvedPredicate>| -> Mark {
                subformulas
                    .lookup_ptr(child)
                    .and_then(|child_sub| self.lookup_node(state, child_sub))
                    .map_or(Mark::Unknown, |dep| self.nodes[dep.0 as usize].mark)
            };

        match formula {
            MuFormula::Atom(_) => None,
            // A Var inherits its binder body's mark (same rule as the
            // worklist-phase `recheck_node`). This is what lets an
            // inner component's resolved value flow into an outer
            // component BEFORE the outer Var's polarity default is
            // applied — e.g. the inner `AG p` True states of
            // `EF(AG p)` becoming μ witnesses, or the inner `EF p`
            // False states of `AG(EF p)` refuting the outer ν.
            MuFormula::Var(v) => {
                let (body, _polarity) = var_env.lookup(*v)?;
                match child_mark(state, body) {
                    Mark::True => Some(Mark::True),
                    Mark::False => Some(Mark::False),
                    Mark::Unknown => None,
                }
            }
            MuFormula::Not(inner) => match child_mark(state, inner) {
                Mark::True => Some(Mark::False),
                Mark::False => Some(Mark::True),
                Mark::Unknown => None,
            },
            MuFormula::And(children) => {
                let mut all_true = true;
                for child in children {
                    match child_mark(state, child) {
                        Mark::False => return Some(Mark::False),
                        Mark::True => {}
                        Mark::Unknown => all_true = false,
                    }
                }
                all_true.then_some(Mark::True)
            }
            MuFormula::Or(children) => {
                let mut all_false = true;
                for child in children {
                    match child_mark(state, child) {
                        Mark::True => return Some(Mark::True),
                        Mark::False => {}
                        Mark::Unknown => all_false = false,
                    }
                }
                all_false.then_some(Mark::False)
            }
            MuFormula::Diamond(inner) => {
                let succs = self.state_space.successors_cached(state)?;
                let mut all_false = true;
                for &s in succs {
                    match child_mark(s, inner) {
                        Mark::True => return Some(Mark::True),
                        Mark::False => {}
                        Mark::Unknown => all_false = false,
                    }
                }
                all_false.then_some(Mark::False)
            }
            MuFormula::Box(inner) => {
                let succs = self.state_space.successors_cached(state)?;
                let mut all_true = true;
                for &s in succs {
                    match child_mark(s, inner) {
                        Mark::False => return Some(Mark::False),
                        Mark::True => {}
                        Mark::Unknown => all_true = false,
                    }
                }
                all_true.then_some(Mark::True)
            }
            MuFormula::Mu(_, body) | MuFormula::Nu(_, body) => match child_mark(state, body) {
                Mark::True => Some(Mark::True),
                Mark::False => Some(Mark::False),
                Mark::Unknown => None,
            },
        }
    }

    #[cfg(test)]
    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// Convenience entry-point: solve `formula` on `net` from the initial
/// marking with the given exploration config.
pub(crate) fn solve_local_mu(
    net: &PetriNet,
    formula: &MuFormula<crate::resolved_predicate::ResolvedPredicate>,
    config: &ExplorationConfig,
) -> Result<bool, MuAbort> {
    let mut solver = LocalMuSolver::new(net, config);
    solver.solve(formula)
}
