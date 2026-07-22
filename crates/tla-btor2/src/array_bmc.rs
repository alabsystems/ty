// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Lazy-array trace-unrolled BMC lane for bit-blast-*ineligible* BTOR2 array
//! nets (index width > 12 or flat expansion > 8192 bits — exactly the class
//! `cmd_btor2` currently punts to the word-level CHC portfolio).
//!
//! # Why unrolling is the sound home for state arrays
//!
//! The combinational Ackermann pre-pass ([`crate::array_elim`]) is a verified
//! DEAD END for *state* arrays: one fresh variable per read cannot track a
//! time-evolving array (per-step consistency is unsound across steps). In a
//! K-step **unrolling** that failure mode disappears: each step gets its own
//! SSA array *epoch* term, so per-epoch consistency is sound. All array
//! reasoning lives in the unrolled trace formula.
//!
//! # Representation
//!
//! Scalars are duplicated per step into a purely combinational
//! [`Btor2Program`] (per-step input copies, frame-0 nondeterministic states as
//! fresh inputs, `S_{t+1}` defined by the step-`t` copy of each `next`
//! expression). Array-sorted nodes NEVER materialize: they are symbolic
//! term-DAG handles ([`ATerm`]) — roots are frame-0 array states (const-array
//! root when init'd by a scalar constant, nondet root when uninit'd) and
//! per-step epoch terms formed by `write`/`ite` chains. Every `read`
//! instance becomes a FRESH input variable plus a read-table entry
//! (select-as-fresh-variable): a pure OVER-approximation with zero axioms.
//!
//! # Lazy axiom instances (added as `constraint` nodes on refinement demand)
//!
//! 1. **Read-over-write**: `r = read(write(B,wi,wv), j)` gets
//!    `r = ite(j==wi, wv, r')` with `r'` a lazily created read on `B` at `j` —
//!    one chain link per demand, never the whole chain.
//! 2. **Select-over-ite**: a read on `ite(c,a,b)` gets `r = ite(c, r_a, r_b)`
//!    with lazy reads on both branches.
//! 3. **Congruence** (Ackermann pairs) per array TERM:
//!    `(idx_i == idx_j) => (r_i == r_j)`, instantiated pairwise on demand.
//! 4. **Const-array roots**: a read on a root with scalar-init default `d`
//!    gets `r = d`.
//!
//! # Extensionality (array equality — phase 2)
//!
//! `eq`/`neq` whose operands are both array-sorted terms of identical dims
//! (states/writes/ites — everything [`ATerm`] already models) become an
//! [`EqEntry`]: a fresh 1-bit input `e` abstracting `A == B` (`neq` consumes
//! the negated var), deduplicated per unordered term pair. Three lazy axiom
//! forms, chosen by domain size with a threshold DERIVED from the read budget
//! (never a magic constant):
//!
//! * **E1** (small `2^iw`, expanded iff `2·2^iw` fits the remaining read
//!   budget): the full biconditional `e <=> AND_j read(A,j)==read(B,j)` over
//!   lazy reads at constant indices — exact in both polarities, no further
//!   refinement for the entry (the reads themselves still refine lazily).
//! * **E2** (skolem, `a != b` on large domains): fresh `iw`-bit input `k` with
//!   `!e => read(A,k) != read(B,k)` — the sound witness-index
//!   over-approximation of `a != b => EXISTS k. a[k] != b[k]`; one per entry.
//! * **E3** (equal-side propagation, `a == b` on large domains): on demand,
//!   `e => read(A,j) == read(B,j)` at the concrete disagreeing index `j`
//!   (a `constd`), instantiated from `refine_or_extract` when the model's `e`
//!   disagrees with the concrete/completable truth of the chains.
//!
//! When `e = 1` is claimed with a nondeterministic root whose residual domain
//! is unread, `extract_model` COMPLETES the unread cells identically on both
//! sides (default + per-index completion) so the claim replays through the
//! exact extensional `word_eq`; a completion conflict instantiates the missing
//! axiom instead. Arrays touching array-sorted INPUTS stay declined (the whole
//! net declines on any array input), as do mixed-dims/negated-operand
//! equalities.
//!
//! # K-induction over the lazy core (phase 2 — the stepping stone to IC3)
//!
//! [`check_array_kinduction`] adds ONE new query shape on unchanged
//! machinery: base case = the per-depth BMC loop above (real `init`, each
//! depth closed or a replay-gated cex); step case = a second [`Unroller`]
//! with frame 0 fully nondeterministic (`init` ignored), NOT-bad assumed at
//! frames `0..k-1`, bad asserted at frame `k`. The abstraction only
//! over-approximates (every concrete trace extends to an abstract model with
//! all instantiated axioms true), so step-UNSAT is sound. UNBOUNDED SAFE
//! ([`ArrayKindOutcome::ProvedSafe`]) is minted ONLY after every base depth
//! and the step query are re-discharged through the INDEPENDENT LRAT-checked
//! leaf ([`crate::array_cert`]'s disjoint path: ay-sat proof mode +
//! `ay-lrat-check` on the identical CNF) — otherwise the claim DOWNGRADES to
//! a bounded fact. This is the sole, gated relaxation of phase 1's "UNSAT is
//! never a verdict" rule. No simple-path constraints yet (completeness-only).
//!
//! # Soundness (fail-closed; the phase-1 absolute rules)
//!
//! * The unrolled net is array-free, so [`crate::bitblast()`] consumes it with
//!   no index-width cap; the leaf is a **fresh, non-incremental**
//!   `ay-sat` solve per refinement iteration and per depth — the same
//!   per-query-fresh mitigation `ay-chc` itself applies to array CHC, and the
//!   opposite of the constraint-dense incremental mode documented unreliable
//!   in `tla-aiger/src/ic3/config.rs`. A solver SAT claim is **never
//!   trusted**: it only seeds a candidate model.
//! * **SAT** is surfaced ONLY after the candidate [`WordLevelModel`] replays
//!   to a bad state through the existing [`crate::word_replay`] machinery
//!   (all constraints held), and its witness goes through the same
//!   fail-closed [`build_word_level_witness`] serializer the CHC lane uses. A
//!   spurious model is refined; a non-replayable one is declined — never
//!   reported.
//! * **UNSAT** at depth K is a bounded-depth claim only
//!   ([`ArrayBmcOutcome::BoundedNoCex`]); it is NEVER converted to an `unsat`
//!   verdict — the caller falls through to the existing portfolio unchanged.
//!   A false-UNSAT from the solver can therefore at worst suppress a
//!   counterexample for one depth (a coverage loss, never a wrong verdict).
//! * Any decline (unsupported op, array equality, no-`next` state, budget,
//!   iteration cap) returns [`ArrayBmcOutcome::Declined`] and the caller's
//!   default decision tree is unchanged.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::bitblast::{bitblast, BitblastedCircuit};
use crate::types::{Btor2Line, Btor2Node, Btor2Program, Btor2Sort, NodeId};
use crate::witness::Btor2Witness;
use crate::word_replay::{
    build_word_level_witness, replay_collect_bad, InitialState, InputFrame, WordLevelModel,
    WordValue,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Configuration for the lazy-array BMC lane.
#[derive(Debug, Clone)]
pub struct ArrayBmcConfig {
    /// Maximum unrolling depth (number of transitions).
    pub max_depth: usize,
    /// Maximum refinement iterations per depth (axiom-instantiation rounds).
    pub max_refinements_per_depth: usize,
    /// Maximum read-table entries per depth (lazy reads included).
    pub max_reads: usize,
    /// Wall-clock budget for the whole lane; `None` = unbounded.
    pub time_budget: Option<Duration>,
    /// Print per-depth/per-iteration progress to stderr.
    pub verbose: bool,
}

impl Default for ArrayBmcConfig {
    fn default() -> Self {
        ArrayBmcConfig {
            max_depth: 20,
            max_refinements_per_depth: 64,
            max_reads: 512,
            time_budget: None,
            verbose: false,
        }
    }
}

/// Outcome of the lazy-array BMC lane. Only [`ArrayBmcOutcome::Unsafe`] is a
/// verdict, and it is replay-proven by construction; the other two are
/// explicit non-verdicts that leave the caller's default paths unchanged.
#[derive(Debug)]
pub enum ArrayBmcOutcome {
    /// A concrete counterexample was found AND confirmed by forward replay
    /// over the original BTOR2 program (`word_replay`, all constraints held).
    Unsafe {
        /// Frame index at which the bad property fires (minimal — depths are
        /// tried in increasing order).
        depth: usize,
        /// Indices into `program.bad_properties` that fire (from replay, not
        /// from the solver).
        fired: Vec<usize>,
        /// The replay-validated concrete model.
        model: WordLevelModel,
        /// The standard btorsim witness, if the model serializes through the
        /// fail-closed shared serializer (`None` e.g. for a nonzero-default
        /// initial array — the verdict is still replay-proven).
        witness: Option<Btor2Witness>,
    },
    /// No counterexample exists within `depth_reached` transitions (each depth
    /// closed UNSAT on an over-approximation). A BOUNDED claim only — never an
    /// `unsat` verdict; callers must fall through to their existing paths.
    BoundedNoCex {
        /// The last depth that was closed.
        depth_reached: usize,
    },
    /// The net is outside the lane's supported slice, or a budget/iteration
    /// cap was hit. The caller's default decision tree proceeds unchanged.
    Declined {
        /// Why the lane declined.
        reason: String,
    },
}

/// Run the lazy-array BMC lane. See the module docs for the abstraction,
/// refinement, and fail-closed rules.
#[must_use]
pub fn check_array_bmc(program: &Btor2Program, config: &ArrayBmcConfig) -> ArrayBmcOutcome {
    match check_inner(program, config) {
        Ok(outcome) => outcome,
        Err(reason) => ArrayBmcOutcome::Declined { reason },
    }
}

// ---------------------------------------------------------------------------
// Preflight (decline rules)
// ---------------------------------------------------------------------------

pub(crate) fn mask(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

pub(crate) fn array_dims(sort: &Btor2Sort) -> Option<(u32, u32)> {
    let Btor2Sort::Array { index, element } = sort else {
        return None;
    };
    match (index.as_ref(), element.as_ref()) {
        (Btor2Sort::BitVec(iw), Btor2Sort::BitVec(ew)) => Some((*iw, *ew)),
        _ => None,
    }
}

/// True iff every node in `id`'s transitive support is constant (no `input`,
/// no `state`) — i.e. the expression is evaluable at init time, matching the
/// empty-context evaluation `word_replay::build_frame0_state` performs.
fn const_support(line_index: &HashMap<NodeId, &Btor2Line>, id: NodeId) -> bool {
    let mut stack = vec![id.unsigned_abs() as i64];
    let mut seen = HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        let Some(line) = line_index.get(&n) else {
            return false;
        };
        match &line.node {
            Btor2Node::Input(_, _) | Btor2Node::State(_, _) => return false,
            _ => {}
        }
        for &a in &line.args {
            stack.push(a.unsigned_abs() as i64);
        }
    }
    true
}

/// Structural preflight: returns the reason the net is outside the lane's
/// phase-1 slice, or `Ok(())`.
#[allow(clippy::too_many_lines)]
pub(crate) fn lane_supported(program: &Btor2Program) -> Result<(), String> {
    if program.bad_properties.is_empty() {
        return Err("no bad properties".into());
    }

    let line_index: HashMap<NodeId, &Btor2Line> = program.lines.iter().map(|l| (l.id, l)).collect();
    let sort_of = |id: NodeId| -> Option<&Btor2Sort> {
        line_index
            .get(&(id.unsigned_abs() as i64))
            .and_then(|l| program.sorts.get(&l.sort_id))
    };
    let is_array = |id: NodeId| matches!(sort_of(id), Some(Btor2Sort::Array { .. }));

    let mut has_array_state = false;
    let mut next_of: HashSet<NodeId> = HashSet::new();
    let mut init_of: HashMap<NodeId, NodeId> = HashMap::new();
    for line in &program.lines {
        match &line.node {
            Btor2Node::Next(_, sid, _) => {
                next_of.insert(sid.unsigned_abs() as i64);
            }
            Btor2Node::Init(_, sid, vid) => {
                init_of.insert(sid.unsigned_abs() as i64, *vid);
            }
            _ => {}
        }
    }

    for line in &program.lines {
        let line_sort = program.sorts.get(&line.sort_id);
        match &line.node {
            Btor2Node::State(_, _) => {
                match line_sort {
                    Some(s @ Btor2Sort::Array { .. }) => {
                        if array_dims(s).is_none() {
                            return Err("array state with non-bitvector index/element".into());
                        }
                        has_array_state = true;
                        // Array init must be a plain scalar constant (the
                        // const-array broadcast idiom) — anything else is
                        // phase 2.
                        if let Some(&vid) = init_of.get(&line.id) {
                            let vid_abs = vid.unsigned_abs() as i64;
                            let ok = matches!(
                                line_index.get(&vid_abs).map(|l| &l.node),
                                Some(
                                    Btor2Node::Zero
                                        | Btor2Node::One
                                        | Btor2Node::Ones
                                        | Btor2Node::Const(_)
                                        | Btor2Node::ConstD(_)
                                        | Btor2Node::ConstH(_)
                                )
                            ) && vid > 0;
                            if !ok {
                                return Err(format!(
                                    "array state {} has a non-constant-scalar init (phase-2)",
                                    line.id
                                ));
                            }
                        }
                    }
                    Some(Btor2Sort::BitVec(_)) => {
                        if let Some(&vid) = init_of.get(&line.id) {
                            if !const_support(&line_index, vid) {
                                return Err(format!(
                                    "scalar state {} has a non-constant init expression",
                                    line.id
                                ));
                            }
                        }
                    }
                    None => return Err(format!("state {} has undefined sort", line.id)),
                }
                // DECLINE any state with no `next`: the lanes disagree on its
                // semantics (hold vs havoc — see
                // docs/hwmcc/no-next-havoc-vs-hold-divergence.md); this lane
                // refuses to pick a side.
                if !next_of.contains(&line.id) {
                    return Err(format!(
                        "state {} has no `next` line (havoc-vs-hold divergence — declined)",
                        line.id
                    ));
                }
            }
            Btor2Node::Input(_, _) => {
                if matches!(line_sort, Some(Btor2Sort::Array { .. })) {
                    return Err("array-sorted input (re-havocked per step — phase 2)".into());
                }
            }
            // Extensional array equality (phase 2): both operands must be
            // non-negated array terms of IDENTICAL dims — everything the
            // epoch term-DAG models. Anything else stays declined.
            Btor2Node::Eq | Btor2Node::Neq => {
                let n_arr = line.args.iter().filter(|&&a| is_array(a)).count();
                if n_arr > 0 {
                    if line.args.len() != 2 || n_arr != 2 {
                        return Err("array-vs-scalar equality — declined".into());
                    }
                    if line.args.iter().any(|&a| a < 0) {
                        return Err("negated array operand in equality — declined".into());
                    }
                    let da = sort_of(line.args[0]).and_then(array_dims);
                    let db = sort_of(line.args[1]).and_then(array_dims);
                    match (da, db) {
                        (Some(x), Some(y)) if x == y => {}
                        (Some(_), Some(_)) => {
                            return Err("mixed-dims array equality — declined".into());
                        }
                        _ => {
                            return Err("array equality with non-bitvector dims — declined".into());
                        }
                    }
                }
            }
            _ => {}
        }

        // Array-escape rule: an array-sorted node may be referenced only where
        // the unroller models it (read/write bases, array-ite branches,
        // init/next of array states). Anything else (concat, output, shift
        // amounts, ...) is outside the slice.
        for (pos, &arg) in line.args.iter().enumerate() {
            if !is_array(arg) {
                continue;
            }
            if arg < 0 {
                return Err("negated array operand".into());
            }
            let ok = match &line.node {
                Btor2Node::Read | Btor2Node::Write => pos == 0,
                Btor2Node::Ite => {
                    (pos == 1 || pos == 2) && matches!(line_sort, Some(Btor2Sort::Array { .. }))
                }
                Btor2Node::Init(_, _, _) | Btor2Node::Next(_, _, _) => true,
                // Validated above: both operands are same-dims arrays.
                Btor2Node::Eq | Btor2Node::Neq => true,
                _ => false,
            };
            if !ok {
                return Err(format!(
                    "array node {arg} escapes into unsupported context ({:?})",
                    line.node
                ));
            }
        }
    }

    if !has_array_state {
        return Err("no array state — not this lane's class".into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The unrolled trace formula
// ---------------------------------------------------------------------------

/// A symbolic array epoch term. Terms never materialize in the unrolled net;
/// they exist only to drive lazy axiom instantiation and concrete chain
/// resolution during refinement.
#[derive(Clone, Copy)]
pub(crate) enum ATerm {
    /// Frame-0 array state initialized to the constant `default` in every cell.
    RootInit { iw: u32, ew: u32, default: u128 },
    /// Frame-0 array state with no `init` — contents nondeterministic.
    RootNondet { state_id: NodeId, iw: u32, ew: u32 },
    /// `write(base, idx, val)` — `idx_u`/`val_u` are unrolled scalar node ids;
    /// `idx_role`/`val_role` index mirror inputs carrying their model values.
    Write {
        base: usize,
        iw: u32,
        ew: u32,
        idx_u: i64,
        val_u: i64,
        idx_role: usize,
        val_role: usize,
    },
    /// `ite(cond, then_t, else_t)` over arrays.
    Ite {
        then_t: usize,
        else_t: usize,
        iw: u32,
        ew: u32,
        cond_u: i64,
        cond_role: usize,
    },
}

impl ATerm {
    pub(crate) fn dims(&self) -> (u32, u32) {
        match self {
            ATerm::RootInit { iw, ew, .. }
            | ATerm::RootNondet { iw, ew, .. }
            | ATerm::Write { iw, ew, .. }
            | ATerm::Ite { iw, ew, .. } => (*iw, *ew),
        }
    }
}

/// One `read` instance: a fresh input variable `var_u` in the unrolled net
/// plus the (term, index) it abstracts.
pub(crate) struct ReadEntry {
    pub(crate) term: usize,
    pub(crate) idx_u: i64,
    pub(crate) var_u: i64,
    pub(crate) idx_role: usize,
    pub(crate) var_role: usize,
    /// Whether this read's structural axiom (ROW / select-over-ite /
    /// const-root) has been instantiated.
    pub(crate) axiom_done: bool,
}

/// One array-equality instance `A == B`: a fresh 1-bit input `var_u`
/// abstracting the extensional truth (deduplicated per unordered term pair;
/// `neq` consumes the negated var). Axioms E1/E2/E3 attach lazily.
pub(crate) struct EqEntry {
    term_a: usize,
    term_b: usize,
    /// The fresh 1-bit equality variable in the unrolled net.
    var_u: i64,
    /// Role index of `var_u` (its concrete value in a model).
    var_role: usize,
    iw: u32,
    ew: u32,
    /// E2 witness-index skolem `(k_u node, k role)` — at most one, ever.
    skolem: Option<(i64, usize)>,
    /// E1: the full small-domain biconditional has been emitted (exact both
    /// polarities — the entry itself needs no further refinement).
    expanded: bool,
    /// E3 instances already emitted (concrete indices).
    e3_done: HashSet<u128>,
}

/// What each unrolled input carries — the key to decoding a SAT model back
/// into a word-level candidate without evaluating any expression.
pub(crate) enum InputRole {
    /// Copy of a source `input` at a frame.
    FrameInput { frame: usize, src: NodeId },
    /// Frame-0 value of a no-`init` scalar state.
    Frame0State { src: NodeId },
    /// A read's fresh value variable.
    ReadVar,
    /// A mirror input constrained equal to an unrolled expression (write
    /// index/value, ite condition, read index) so its concrete value is
    /// directly readable from the model.
    Mirror,
    /// An array-equality entry's fresh 1-bit truth variable.
    EqVar,
    /// An E2 witness-index skolem input (`iw` bits, otherwise unconstrained).
    EqSkolem,
    /// A free, unconstrained index input (Λ-pin universal-cell index in the
    /// IC3 lane; existential witness index in the Tier-B validator). Like
    /// `EqSkolem`, it never decodes into a word-level model.
    FreeIndex,
}

pub(crate) struct Unroller<'a> {
    program: &'a Btor2Program,
    line_index: HashMap<NodeId, &'a Btor2Line>,
    /// state id -> (init value node, next value node).
    init_of: HashMap<NodeId, NodeId>,
    next_of: HashMap<NodeId, NodeId>,
    pub(crate) state_ids: Vec<NodeId>,

    // -- the unrolled combinational program (array-free) ---------------------
    lines: Vec<Btor2Line>,
    sorts: HashMap<i64, Btor2Sort>,
    sort_cache: HashMap<u32, i64>,
    constraints_ids: Vec<i64>,
    next_id: i64,
    num_inputs: usize,

    // -- per-(frame, source-node) memoization ---------------------------------
    scalar_map: HashMap<(usize, NodeId), i64>,
    array_map: HashMap<(usize, NodeId), usize>,
    /// Highest frame whose state values are seeded.
    max_frame: usize,
    /// Highest frame whose constraints have been emitted.
    constraints_emitted_through: Option<usize>,

    // -- lazy-array bookkeeping -----------------------------------------------
    pub(crate) terms: Vec<ATerm>,
    pub(crate) reads: Vec<ReadEntry>,
    read_dedup: HashMap<(usize, i64), usize>,
    congruence_pairs: HashSet<(usize, usize)>,
    pub(crate) input_roles: Vec<InputRole>,

    // -- extensionality bookkeeping --------------------------------------------
    pub(crate) eqs: Vec<EqEntry>,
    /// Unordered (term, term) -> eq entry index.
    eq_dedup: HashMap<(usize, usize), usize>,
    /// Cache of emitted `constd` index nodes per (width, value), so E1/E3
    /// reads at the same constant index dedup through `read_dedup`.
    const_cache: HashMap<(u32, u128), i64>,
    /// The caller's read budget (`config.max_reads`) — the E1 expansion
    /// threshold is DERIVED from it: expand iff `2 * 2^iw` fits the remaining
    /// budget.
    read_budget: usize,
    /// K-induction STEP mode: frame 0 is fully nondeterministic (`init` lines
    /// ignored — fresh inputs for every scalar state, `RootNondet` for every
    /// array state). Sound for UNSAT-side reasoning only: the step
    /// abstraction over-approximates every reachable suffix.
    free_init: bool,
}

impl<'a> Unroller<'a> {
    pub(crate) fn new(
        program: &'a Btor2Program,
        read_budget: usize,
        free_init: bool,
    ) -> Result<Self, String> {
        let line_index: HashMap<NodeId, &Btor2Line> =
            program.lines.iter().map(|l| (l.id, l)).collect();
        let mut init_of = HashMap::new();
        let mut next_of = HashMap::new();
        let mut state_ids = Vec::new();
        for line in &program.lines {
            match &line.node {
                Btor2Node::Init(_, sid, vid) => {
                    init_of.insert(sid.unsigned_abs() as i64, *vid);
                }
                Btor2Node::Next(_, sid, vid) => {
                    next_of.insert(sid.unsigned_abs() as i64, *vid);
                }
                Btor2Node::State(_, _) => state_ids.push(line.id),
                _ => {}
            }
        }
        let mut u = Unroller {
            program,
            line_index,
            init_of,
            next_of,
            state_ids,
            lines: Vec::new(),
            sorts: HashMap::new(),
            sort_cache: HashMap::new(),
            constraints_ids: Vec::new(),
            next_id: 1,
            num_inputs: 0,
            scalar_map: HashMap::new(),
            array_map: HashMap::new(),
            max_frame: 0,
            constraints_emitted_through: None,
            terms: Vec::new(),
            reads: Vec::new(),
            read_dedup: HashMap::new(),
            congruence_pairs: HashSet::new(),
            input_roles: Vec::new(),
            eqs: Vec::new(),
            eq_dedup: HashMap::new(),
            const_cache: HashMap::new(),
            read_budget,
            free_init,
        };
        u.seed_frame0()?;
        Ok(u)
    }

    // -- low-level emission ----------------------------------------------------

    fn fresh_id(&mut self) -> i64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn sort_for(&mut self, width: u32) -> i64 {
        if let Some(&id) = self.sort_cache.get(&width) {
            return id;
        }
        let id = self.fresh_id();
        self.lines.push(Btor2Line {
            id,
            sort_id: id,
            node: Btor2Node::SortBitVec(width),
            args: vec![],
        });
        self.sorts.insert(id, Btor2Sort::BitVec(width));
        self.sort_cache.insert(width, id);
        id
    }

    pub(crate) fn emit(&mut self, node: Btor2Node, width: u32, args: Vec<i64>) -> i64 {
        let sort_id = self.sort_for(width);
        let id = self.fresh_id();
        self.lines.push(Btor2Line {
            id,
            sort_id,
            node,
            args,
        });
        id
    }

    fn fresh_input(&mut self, width: u32, name: String, role: InputRole) -> i64 {
        let sort_id = self.sort_for(width);
        let id = self.fresh_id();
        self.lines.push(Btor2Line {
            id,
            sort_id,
            node: Btor2Node::Input(sort_id, Some(name)),
            args: vec![],
        });
        self.num_inputs += 1;
        self.input_roles.push(role);
        id
    }

    /// Fresh UNCONSTRAINED input of `width` bits (a free variable of the
    /// unrolled net): the Λ-pin universal-cell index / Tier-B witness index.
    /// Returns `(node id, input role index)`.
    pub(crate) fn fresh_free_index(&mut self, width: u32, tag: &str) -> (i64, usize) {
        let role_idx = self.input_roles.len();
        let id = self.fresh_input(
            width,
            format!("u_free_{tag}_{role_idx}"),
            InputRole::FreeIndex,
        );
        (id, role_idx)
    }

    pub(crate) fn emit_constraint(&mut self, cond_id: i64) {
        let id = self.fresh_id();
        self.lines.push(Btor2Line {
            id,
            sort_id: 0,
            node: Btor2Node::Constraint(cond_id),
            args: vec![cond_id],
        });
        self.constraints_ids.push(id);
    }

    /// Fresh mirror input `m` with the constraint `m == expr_u`; returns
    /// `(m, role_index)`. Any SAT model must therefore assign `m` the concrete
    /// value of `expr_u`, making that value directly readable from the model.
    pub(crate) fn add_mirror(&mut self, expr_u: i64, width: u32) -> (i64, usize) {
        let role_idx = self.input_roles.len();
        let m = self.fresh_input(width, format!("u_mir_{}", role_idx), InputRole::Mirror);
        let eq = self.emit(Btor2Node::Eq, 1, vec![m, expr_u]);
        self.emit_constraint(eq);
        (m, role_idx)
    }

    // -- source-program helpers --------------------------------------------------

    fn src_line(&self, id: NodeId) -> Result<&'a Btor2Line, String> {
        self.line_index
            .get(&id)
            .copied()
            .ok_or_else(|| format!("undefined source node {id}"))
    }

    fn src_scalar_width(&self, line: &Btor2Line) -> Result<u32, String> {
        match self.program.sorts.get(&line.sort_id) {
            Some(Btor2Sort::BitVec(w)) => Ok(*w),
            Some(Btor2Sort::Array { .. }) => {
                Err(format!("node {} is array-sorted, expected scalar", line.id))
            }
            None => Err(format!("node {} has no sort", line.id)),
        }
    }

    fn src_is_array(&self, id: NodeId) -> bool {
        self.line_index
            .get(&(id.unsigned_abs() as i64))
            .and_then(|l| self.program.sorts.get(&l.sort_id))
            .is_some_and(|s| matches!(s, Btor2Sort::Array { .. }))
    }

    pub(crate) fn node_cond(&self, prop_line: NodeId) -> Result<NodeId, String> {
        match self.src_line(prop_line)?.node {
            Btor2Node::Bad(c) | Btor2Node::Constraint(c) => Ok(c),
            _ => Err(format!("node {prop_line} is not a bad/constraint")),
        }
    }

    // -- frame seeding -------------------------------------------------------------

    fn seed_frame0(&mut self) -> Result<(), String> {
        for i in 0..self.state_ids.len() {
            let sid = self.state_ids[i];
            let line = self.src_line(sid)?;
            let sort = self
                .program
                .sorts
                .get(&line.sort_id)
                .ok_or("state sort missing")?;
            match sort {
                Btor2Sort::BitVec(w) => {
                    // K-induction step mode ignores `init`: frame 0 is fully
                    // nondeterministic.
                    let init = if self.free_init {
                        None
                    } else {
                        self.init_of.get(&sid).copied()
                    };
                    let u = match init {
                        // Const-support checked in preflight: unrolling the
                        // init expr at frame 0 yields a constant subcircuit.
                        Some(vid) => self.unroll_ref(0, vid)?,
                        None => {
                            let w = *w;
                            self.fresh_input(
                                w,
                                format!("u_st0_{sid}"),
                                InputRole::Frame0State { src: sid },
                            )
                        }
                    };
                    self.scalar_map.insert((0, sid), u);
                }
                s @ Btor2Sort::Array { .. } => {
                    let (iw, ew) = array_dims(s).ok_or("bad array dims")?;
                    let init = if self.free_init {
                        None
                    } else {
                        self.init_of.get(&sid).copied()
                    };
                    let term = match init {
                        Some(vid) => {
                            let default = self.const_scalar_value(vid, ew)?;
                            ATerm::RootInit { iw, ew, default }
                        }
                        None => ATerm::RootNondet {
                            state_id: sid,
                            iw,
                            ew,
                        },
                    };
                    self.terms.push(term);
                    self.array_map.insert((0, sid), self.terms.len() - 1);
                }
            }
        }
        Ok(())
    }

    /// Concrete value of a constant scalar node (preflight guarantees the
    /// shape), masked to `width`.
    fn const_scalar_value(&self, id: NodeId, width: u32) -> Result<u128, String> {
        let line = self.src_line(id.unsigned_abs() as i64)?;
        let v = match &line.node {
            Btor2Node::Zero => 0,
            Btor2Node::One => 1,
            Btor2Node::Ones => mask(width),
            Btor2Node::Const(s) => u128::from_str_radix(s, 2).map_err(|e| format!("const: {e}"))?,
            Btor2Node::ConstD(s) => {
                if let Some(stripped) = s.strip_prefix('-') {
                    stripped
                        .parse::<u128>()
                        .map_err(|e| format!("constd: {e}"))?
                        .wrapping_neg()
                } else {
                    s.parse::<u128>().map_err(|e| format!("constd: {e}"))?
                }
            }
            Btor2Node::ConstH(s) => {
                u128::from_str_radix(s, 16).map_err(|e| format!("consth: {e}"))?
            }
            other => return Err(format!("expected constant scalar, got {other:?}")),
        };
        Ok(v & mask(width))
    }

    /// Advance seeded state values through `frame` (simultaneous commit: each
    /// step's `next` expressions are unrolled against the previous frame).
    pub(crate) fn seed_through(&mut self, frame: usize) -> Result<(), String> {
        while self.max_frame < frame {
            let f = self.max_frame;
            for i in 0..self.state_ids.len() {
                let sid = self.state_ids[i];
                let vid = *self
                    .next_of
                    .get(&sid)
                    .ok_or_else(|| format!("state {sid} has no next"))?;
                if self.src_is_array(sid) {
                    let t = self.resolve_array(f, vid)?;
                    self.array_map.insert((f + 1, sid), t);
                } else {
                    let u = self.unroll_ref(f, vid)?;
                    self.scalar_map.insert((f + 1, sid), u);
                }
            }
            self.max_frame += 1;
        }
        Ok(())
    }

    /// Emit the per-frame copies of every `constraint` through `frame`.
    pub(crate) fn emit_constraints_through(&mut self, frame: usize) -> Result<(), String> {
        let start = match self.constraints_emitted_through {
            None => 0,
            Some(done) => done + 1,
        };
        for f in start..=frame {
            self.seed_through(f)?;
            for i in 0..self.program.constraints.len() {
                let cid = self.program.constraints[i];
                let cond = self.node_cond(cid)?;
                let cu = self.unroll_ref(f, cond)?;
                self.emit_constraint(cu);
            }
        }
        self.constraints_emitted_through =
            Some(frame.max(self.constraints_emitted_through.unwrap_or(0)));
        Ok(())
    }

    // -- expression unrolling ---------------------------------------------------

    /// Unroll a (possibly negated) scalar operand reference at `frame`.
    pub(crate) fn unroll_ref(&mut self, frame: usize, r: i64) -> Result<i64, String> {
        let abs = r.unsigned_abs() as i64;
        let u = self.unroll_scalar(frame, abs)?;
        Ok(if r < 0 { -u } else { u })
    }

    #[allow(clippy::too_many_lines)]
    fn unroll_scalar(&mut self, frame: usize, abs: NodeId) -> Result<i64, String> {
        if let Some(&u) = self.scalar_map.get(&(frame, abs)) {
            return Ok(u);
        }
        let line = self.src_line(abs)?;
        let u = match &line.node {
            Btor2Node::State(_, _) => {
                return Err(format!(
                    "state {abs} not seeded at frame {frame} (unroller invariant)"
                ));
            }
            Btor2Node::Input(_, _) => {
                let w = self.src_scalar_width(line)?;
                self.fresh_input(
                    w,
                    format!("u_in_f{frame}_{abs}"),
                    InputRole::FrameInput { frame, src: abs },
                )
            }
            Btor2Node::Read => {
                let term = self.resolve_array(frame, line.args[0])?;
                let idx_u = self.unroll_ref(frame, line.args[1])?;
                let entry = self.get_or_make_read(term, idx_u)?;
                self.reads[entry].var_u
            }
            // Extensional array equality: both operands resolve to epoch
            // terms; the result is the entry's fresh 1-bit truth variable
            // (negated for `neq`). Preflight guarantees the operand shape.
            Btor2Node::Eq | Btor2Node::Neq if line.args.iter().any(|&a| self.src_is_array(a)) => {
                if line.args.len() != 2 || !line.args.iter().all(|&a| a > 0 && self.src_is_array(a))
                {
                    return Err(format!(
                        "unsupported array equality shape at node {abs} (preflight gap)"
                    ));
                }
                let ta = self.resolve_array(frame, line.args[0])?;
                let tb = self.resolve_array(frame, line.args[1])?;
                let entry = self.get_or_make_eq(ta, tb)?;
                let var = self.eqs[entry].var_u;
                if matches!(line.node, Btor2Node::Eq) {
                    var
                } else {
                    -var
                }
            }
            // Scalar ops (including scalar ite): duplicate the line with
            // per-frame operand copies. The bit-blaster supports the full
            // scalar op set, so no per-op whitelist is needed here.
            _ => {
                let w = self.src_scalar_width(line)?;
                let mut args = Vec::with_capacity(line.args.len());
                for &a in &line.args {
                    if self.src_is_array(a) {
                        return Err(format!(
                            "array operand {a} in scalar context ({:?})",
                            line.node
                        ));
                    }
                    args.push(self.unroll_ref(frame, a)?);
                }
                self.emit(line.node.clone(), w, args)
            }
        };
        self.scalar_map.insert((frame, abs), u);
        Ok(u)
    }

    fn resolve_array(&mut self, frame: usize, r: i64) -> Result<usize, String> {
        if r < 0 {
            return Err("negated array operand".into());
        }
        if let Some(&t) = self.array_map.get(&(frame, r)) {
            return Ok(t);
        }
        let line = self.src_line(r)?;
        let sort = self
            .program
            .sorts
            .get(&line.sort_id)
            .ok_or("array sort missing")?;
        let (iw, ew) = array_dims(sort).ok_or("non-bitvector array dims")?;
        let t = match &line.node {
            Btor2Node::State(_, _) => {
                return Err(format!(
                    "array state {r} not seeded at frame {frame} (unroller invariant)"
                ));
            }
            Btor2Node::Write => {
                let base = self.resolve_array(frame, line.args[0])?;
                let idx_u = self.unroll_ref(frame, line.args[1])?;
                let val_u = self.unroll_ref(frame, line.args[2])?;
                let (_, idx_role) = self.add_mirror(idx_u, iw);
                let (_, val_role) = self.add_mirror(val_u, ew);
                self.terms.push(ATerm::Write {
                    base,
                    iw,
                    ew,
                    idx_u,
                    val_u,
                    idx_role,
                    val_role,
                });
                self.terms.len() - 1
            }
            Btor2Node::Ite => {
                let cond_u = self.unroll_ref(frame, line.args[0])?;
                let then_t = self.resolve_array(frame, line.args[1])?;
                let else_t = self.resolve_array(frame, line.args[2])?;
                let (_, cond_role) = self.add_mirror(cond_u, 1);
                self.terms.push(ATerm::Ite {
                    then_t,
                    else_t,
                    iw,
                    ew,
                    cond_u,
                    cond_role,
                });
                self.terms.len() - 1
            }
            other => return Err(format!("unsupported array-producing node {other:?}")),
        };
        self.array_map.insert((frame, r), t);
        Ok(t)
    }

    /// The select-as-fresh-variable core: a read on `(term, idx_u)` becomes a
    /// fresh input plus a read-table entry (deduplicated per (term, idx node)).
    pub(crate) fn get_or_make_read(&mut self, term: usize, idx_u: i64) -> Result<usize, String> {
        if let Some(&i) = self.read_dedup.get(&(term, idx_u)) {
            return Ok(i);
        }
        let (iw, ew) = self.terms[term].dims();
        let read_idx = self.reads.len();
        let var_role = self.input_roles.len();
        let var_u = self.fresh_input(ew, format!("u_rd_{read_idx}"), InputRole::ReadVar);
        let (_, idx_role) = self.add_mirror(idx_u, iw);
        self.reads.push(ReadEntry {
            term,
            idx_u,
            var_u,
            idx_role,
            var_role,
            axiom_done: false,
        });
        self.read_dedup.insert((term, idx_u), read_idx);
        Ok(read_idx)
    }

    /// Cached `constd` node of `val` at `width` (so constant-index reads from
    /// E1/E3 dedup through `read_dedup`).
    pub(crate) fn const_index(&mut self, width: u32, val: u128) -> i64 {
        if let Some(&id) = self.const_cache.get(&(width, val)) {
            return id;
        }
        let id = self.emit(Btor2Node::ConstD(val.to_string()), width, vec![]);
        self.const_cache.insert((width, val), id);
        id
    }

    /// Get or create the [`EqEntry`] for the unordered term pair, expanding
    /// the exact E1 biconditional immediately when the domain fits the
    /// remaining read budget (derived threshold — never a magic constant).
    fn get_or_make_eq(&mut self, ta: usize, tb: usize) -> Result<usize, String> {
        let key = (ta.min(tb), ta.max(tb));
        if let Some(&i) = self.eq_dedup.get(&key) {
            return Ok(i);
        }
        let (iwa, ewa) = self.terms[key.0].dims();
        let (iwb, ewb) = self.terms[key.1].dims();
        if (iwa, ewa) != (iwb, ewb) {
            return Err("mixed-dims array equality".into());
        }
        let ei = self.eqs.len();
        let var_role = self.input_roles.len();
        let var_u = self.fresh_input(1, format!("u_eq_{ei}"), InputRole::EqVar);
        self.eqs.push(EqEntry {
            term_a: key.0,
            term_b: key.1,
            var_u,
            var_role,
            iw: iwa,
            ew: ewa,
            skolem: None,
            expanded: false,
            e3_done: HashSet::new(),
        });
        self.eq_dedup.insert(key, ei);
        // E1 derived threshold: the full biconditional needs one read per
        // side per index — expand iff 2 * 2^iw fits the remaining budget.
        if iwa < 64 {
            let needed = 2u128 << iwa; // 2 * 2^iw
            let remaining = self.read_budget.saturating_sub(self.reads.len());
            if needed <= remaining as u128 {
                self.expand_e1(ei)?;
            }
        }
        Ok(ei)
    }

    /// E1: `e <=> AND_{j < 2^iw} read(A,j) == read(B,j)` over lazy reads at
    /// constant indices. Exact in BOTH polarities; the reads still refine
    /// lazily through the ordinary structural/congruence machinery.
    fn expand_e1(&mut self, ei: usize) -> Result<(), String> {
        let (ta, tb, iw, var_u) = {
            let e = &self.eqs[ei];
            (e.term_a, e.term_b, e.iw, e.var_u)
        };
        let domain = 1u128 << iw;
        let mut conj: Option<i64> = None;
        for j in 0..domain {
            let idx = self.const_index(iw, j);
            let ra = self.get_or_make_read(ta, idx)?;
            let rb = self.get_or_make_read(tb, idx)?;
            let (va, vb) = (self.reads[ra].var_u, self.reads[rb].var_u);
            let eq_j = self.emit(Btor2Node::Eq, 1, vec![va, vb]);
            conj = Some(match conj {
                None => eq_j,
                Some(c) => self.emit(Btor2Node::And, 1, vec![c, eq_j]),
            });
        }
        let c = conj.ok_or("empty array domain in E1 expansion")?;
        let bicond = self.emit(Btor2Node::Eq, 1, vec![var_u, c]);
        self.emit_constraint(bicond);
        self.eqs[ei].expanded = true;
        Ok(())
    }

    /// E2: one skolem witness index per entry, ever:
    /// `!e => read(A,k) != read(B,k)` with `k` a fresh unconstrained input —
    /// the sound over-approximation of `a != b => EXISTS k. a[k] != b[k]`.
    /// Returns `true` iff newly instantiated.
    pub(crate) fn instantiate_skolem(&mut self, ei: usize) -> bool {
        if self.eqs[ei].skolem.is_some() {
            return false;
        }
        let (ta, tb, iw, var_u) = {
            let e = &self.eqs[ei];
            (e.term_a, e.term_b, e.iw, e.var_u)
        };
        let k_role = self.input_roles.len();
        let k_u = self.fresh_input(iw, format!("u_eqk_{ei}"), InputRole::EqSkolem);
        let (Ok(ra), Ok(rb)) = (
            self.get_or_make_read(ta, k_u),
            self.get_or_make_read(tb, k_u),
        ) else {
            return false;
        };
        let (va, vb) = (self.reads[ra].var_u, self.reads[rb].var_u);
        let neq = self.emit(Btor2Node::Neq, 1, vec![va, vb]);
        let imp = self.emit(Btor2Node::Implies, 1, vec![-var_u, neq]);
        self.emit_constraint(imp);
        self.eqs[ei].skolem = Some((k_u, k_role));
        true
    }

    /// E3: equal-side read propagation at a concrete index:
    /// `e => read(A,j) == read(B,j)` with `j` emitted as a `constd`. Returns
    /// `true` iff newly instantiated for this (entry, index).
    fn instantiate_e3(&mut self, ei: usize, j: u128) -> bool {
        if self.eqs[ei].e3_done.contains(&j) {
            return false;
        }
        let (ta, tb, iw, var_u) = {
            let e = &self.eqs[ei];
            (e.term_a, e.term_b, e.iw, e.var_u)
        };
        let idx = self.const_index(iw, j & mask(iw));
        let (Ok(ra), Ok(rb)) = (
            self.get_or_make_read(ta, idx),
            self.get_or_make_read(tb, idx),
        ) else {
            return false;
        };
        let (va, vb) = (self.reads[ra].var_u, self.reads[rb].var_u);
        let eq_v = self.emit(Btor2Node::Eq, 1, vec![va, vb]);
        let imp = self.emit(Btor2Node::Implies, 1, vec![var_u, eq_v]);
        self.emit_constraint(imp);
        self.eqs[ei].e3_done.insert(j);
        true
    }

    // -- query assembly -----------------------------------------------------------

    /// Materialize the depth-`t` query: the shared unrolled prefix plus one
    /// `bad` line per property, all at frame `t`. Constraints cover frames
    /// `0..=t` plus every instantiated axiom.
    pub(crate) fn build_query(&mut self, t: usize) -> Result<Btor2Program, String> {
        self.seed_through(t)?;
        self.emit_constraints_through(t)?;

        // Unroll bad conditions at frame t (nodes persist in self.lines; the
        // Bad lines themselves are query-local).
        let mut bad_conds = Vec::new();
        for i in 0..self.program.bad_properties.len() {
            let bid = self.program.bad_properties[i];
            let cond = self.node_cond(bid)?;
            bad_conds.push(self.unroll_ref(t, cond)?);
        }

        Ok(self.assemble_program(&bad_conds))
    }

    /// Materialize the current unrolled prefix as a combinational
    /// [`Btor2Program`] with one query-local `Bad` line per condition id in
    /// `bad_conds` (negated references allowed). `self.lines` is unchanged —
    /// the Bad lines exist only in the returned program.
    pub(crate) fn assemble_program(&self, bad_conds: &[i64]) -> Btor2Program {
        let mut lines = self.lines.clone();
        let mut bad_ids = Vec::new();
        let mut next_id = self.next_id;
        for &bu in bad_conds {
            let id = next_id;
            next_id += 1;
            lines.push(Btor2Line {
                id,
                sort_id: 0,
                node: Btor2Node::Bad(bu),
                args: vec![bu],
            });
            bad_ids.push(id);
        }
        Btor2Program {
            lines,
            sorts: self.sorts.clone(),
            num_inputs: self.num_inputs,
            num_states: 0,
            bad_properties: bad_ids,
            constraints: self.constraints_ids.clone(),
            fairness: vec![],
            justice: vec![],
        }
    }

    /// Number of unrolled lines (half of the circuit version stamp).
    pub(crate) fn lines_len(&self) -> usize {
        self.lines.len()
    }

    /// Number of emitted constraints (the other half of the version stamp).
    pub(crate) fn constraints_len(&self) -> usize {
        self.constraints_ids.len()
    }

    /// Unrolled node id of scalar state `sid` at `frame` (seeded frames only).
    pub(crate) fn scalar_state_at(&self, frame: usize, sid: NodeId) -> Option<i64> {
        self.scalar_map.get(&(frame, sid)).copied()
    }

    /// Epoch term index of array state `sid` at `frame` (seeded frames only).
    pub(crate) fn array_state_term(&self, frame: usize, sid: NodeId) -> Option<usize> {
        self.array_map.get(&(frame, sid)).copied()
    }
}

// ---------------------------------------------------------------------------
// SAT leaf (fresh, non-incremental per query)
// ---------------------------------------------------------------------------

pub(crate) enum QueryResult {
    Unsat,
    Sat(Vec<bool>),
}

pub(crate) fn lit_to_dimacs(lit: u64) -> i32 {
    let var = (lit >> 1) as i32;
    if lit & 1 == 1 {
        -var
    } else {
        var
    }
}

/// CNF of the combinational query `constraints AND (OR bads)` over the
/// bit-blasted circuit. `None` = structurally UNSAT (a constant-false
/// constraint, or every bad folded to constant false) — sound with no solver
/// involved. Shared between the primary fresh-solver leaf ([`solve_query`])
/// and the LRAT-checked independent discharge ([`discharge_unsat_lrat`]), so
/// both paths see the identical CNF.
pub(crate) fn circuit_to_cnf(circuit: &BitblastedCircuit) -> Option<(usize, Vec<Vec<i32>>)> {
    debug_assert!(
        circuit.latches.is_empty(),
        "unrolled query must be combinational"
    );

    let mut clauses: Vec<Vec<i32>> =
        Vec::with_capacity(circuit.ands.len() * 3 + circuit.constraints.len() + 1);
    for &(lhs, a, b) in &circuit.ands {
        let l = lit_to_dimacs(lhs);
        let la = lit_to_dimacs(a);
        let lb = lit_to_dimacs(b);
        clauses.push(vec![-l, la]);
        clauses.push(vec![-l, lb]);
        clauses.push(vec![l, -la, -lb]);
    }
    for &c in &circuit.constraints {
        match c {
            1 => {}
            0 => return None, // constant-false constraint
            _ => clauses.push(vec![lit_to_dimacs(c)]),
        }
    }
    let mut bad_clause = Vec::new();
    let mut bad_trivially_true = false;
    for &b in &circuit.bad {
        match b {
            0 => {}
            1 => bad_trivially_true = true,
            _ => bad_clause.push(lit_to_dimacs(b)),
        }
    }
    if !bad_trivially_true {
        if bad_clause.is_empty() {
            return None; // every bad folded to constant false
        }
        clauses.push(bad_clause);
    }
    Some((circuit.max_var.max(1) as usize, clauses))
}

/// Solve the combinational query: is there an input assignment with every
/// constraint true and SOME bad literal true? Fresh solver per call — the
/// instantiated-axiom formulas are constraint-dense, precisely the class where
/// ay-sat incremental mode is documented unreliable, so we mirror ay-chc's own
/// per-query-fresh-executor mitigation.
pub(crate) fn solve_query(circuit: &BitblastedCircuit) -> Result<QueryResult, String> {
    let Some((num_vars, clauses)) = circuit_to_cnf(circuit) else {
        return Ok(QueryResult::Unsat);
    };

    let mut dimacs = String::with_capacity(clauses.len() * 12 + 32);
    use std::fmt::Write as _;
    let _ = writeln!(dimacs, "p cnf {} {}", num_vars, clauses.len());
    for cl in &clauses {
        for lit in cl {
            let _ = write!(dimacs, "{lit} ");
        }
        dimacs.push_str("0\n");
    }

    let formula =
        ay_sat::parse_dimacs(&dimacs).map_err(|e| format!("internal CNF parse error: {e:?}"))?;
    let portfolio = ay_sat::PortfolioSolver::new(1);
    match portfolio.solve(&formula) {
        ay_sat::SatResult::Sat(model) => Ok(QueryResult::Sat(model)),
        ay_sat::SatResult::Unsat(_) => Ok(QueryResult::Unsat),
        // `SatResult` is non-exhaustive; anything not a definitive SAT/UNSAT
        // is inconclusive — decline, never guess.
        _ => Err("ay-sat returned an inconclusive result".into()),
    }
}

/// The INDEPENDENT second trust path for a claimed-UNSAT query (the Gate-B /
/// `validate.rs` discipline): rebuild the identical CNF, solve it with ay-sat
/// in proof mode, and accept ONLY an UNSAT whose LRAT proof re-verifies under
/// the separate `ay-lrat-check` crate ([`crate::array_cert`]'s disjoint
/// leaf). Any SAT / inconclusive / unverifiable-proof outcome is an error —
/// the caller must downgrade its claim, never report it.
pub(crate) fn discharge_unsat_lrat(circuit: &BitblastedCircuit) -> Result<(), String> {
    let Some((num_vars, clauses)) = circuit_to_cnf(circuit) else {
        // Structurally UNSAT (constant-false constraint / no bad literal):
        // sound without a solver.
        return Ok(());
    };
    let cnf = crate::array_cert::DimacsCnf {
        num_vars,
        clauses,
        trivially_unsat: false,
    };
    // Same discipline as the Gate-B VC leaf: cap proof bookkeeping with the
    // shared resource-derived byte budget so a runaway proof construction
    // degrades to the fail-closed Inconclusive arm instead of running
    // unbounded.
    match crate::array_cert::solve_dimacs_unsat_lrat(
        &cnf,
        Some(crate::array_cert::derive_byte_budget()),
    ) {
        crate::array_cert::LeafOutcome::VerifiedUnsat => Ok(()),
        crate::array_cert::LeafOutcome::Sat => {
            Err("independent discharge found the query SAT".into())
        }
        crate::array_cert::LeafOutcome::Inconclusive(r) => {
            Err(format!("independent discharge inconclusive: {r}"))
        }
    }
}

/// Decode the concrete value of every unrolled input from the SAT model,
/// aligned with `input_roles` (bit-blast emits `input_bits` in program order,
/// which is the unroller's emission order).
pub(crate) fn decode_input_values(circuit: &BitblastedCircuit, model: &[bool]) -> Vec<u128> {
    circuit
        .input_bits
        .iter()
        .map(|(_, bits)| {
            let mut v: u128 = 0;
            for (j, &lit) in bits.iter().enumerate() {
                let b = match lit {
                    0 => false,
                    1 => true,
                    _ => {
                        let var = (lit >> 1) as usize;
                        model.get(var.wrapping_sub(1)).copied().unwrap_or(false)
                    }
                };
                if b && j < 128 {
                    v |= 1u128 << j;
                }
            }
            v
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Refinement (spurious-model detection + lazy axiom instantiation)
// ---------------------------------------------------------------------------

/// Concrete resolution target of a read under the model.
enum Resolved {
    /// Defined by a write's stored value.
    ByWrite(u128),
    /// Lands on a const-init root: value is the root default.
    ByInitRoot(u128),
    /// Lands on a nondet root at a concrete cell index.
    NondetCell { term: usize, idx: u128 },
}

fn resolve_concrete(terms: &[ATerm], mut term: usize, idx: u128, vals: &[u128]) -> Resolved {
    loop {
        match &terms[term] {
            ATerm::Write {
                base,
                iw,
                idx_role,
                val_role,
                ..
            } => {
                if vals[*idx_role] & mask(*iw) == idx {
                    return Resolved::ByWrite(vals[*val_role]);
                }
                term = *base;
            }
            ATerm::Ite {
                then_t,
                else_t,
                cond_role,
                ..
            } => {
                term = if vals[*cond_role] & 1 == 1 {
                    *then_t
                } else {
                    *else_t
                };
            }
            ATerm::RootInit { default, .. } => return Resolved::ByInitRoot(*default),
            ATerm::RootNondet { .. } => return Resolved::NondetCell { term, idx },
        }
    }
}

/// The chain-consistent candidate assignment handed to `extract_model`: the
/// nondet-root cells resolved from the model, plus the extensionality
/// COMPLETION (extra cells and per-root defaults) that makes every claimed
/// `e = 1` replay through the exact extensional `word_eq`.
pub(crate) struct CandidatePlan {
    /// Nondet root term -> (cell index -> value), from read resolution plus
    /// eq completion.
    pub(crate) root_cells: HashMap<usize, HashMap<u128, u128>>,
    /// Nondet root term -> completed residual default (absent = 0).
    pub(crate) root_defaults: HashMap<usize, u128>,
}

/// Root of a model-concretized epoch chain.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EffRoot {
    /// Const-init root: every unwritten index reads this default.
    Init(u128),
    /// Nondet root term index: unwritten, unread indices are FREE.
    Nondet(usize),
}

/// Concretize an epoch chain under the model: walk model-selected `ite`
/// branches and collect the outermost write per concrete index; return the
/// root and the override map. Exactly mirrors `resolve_concrete`, but for the
/// WHOLE index domain at once.
fn effective_array(terms: &[ATerm], mut t: usize, vals: &[u128]) -> (EffRoot, HashMap<u128, u128>) {
    let mut overrides: HashMap<u128, u128> = HashMap::new();
    loop {
        match &terms[t] {
            ATerm::Write {
                base,
                iw,
                ew,
                idx_role,
                val_role,
                ..
            } => {
                let i = vals[*idx_role] & mask(*iw);
                overrides.entry(i).or_insert(vals[*val_role] & mask(*ew));
                t = *base;
            }
            ATerm::Ite {
                then_t,
                else_t,
                cond_role,
                ..
            } => {
                t = if vals[*cond_role] & 1 == 1 {
                    *then_t
                } else {
                    *else_t
                };
            }
            ATerm::RootInit { default, ew, .. } => {
                return (EffRoot::Init(*default & mask(*ew)), overrides)
            }
            ATerm::RootNondet { .. } => return (EffRoot::Nondet(t), overrides),
        }
    }
}

/// One refinement pass over the read table (then the eq table). Returns the
/// candidate assignment if the model is chain-consistent AND every array
/// equality is consistent/completable, or instantiates missing axioms
/// (mutating `u`) and returns `Err(axioms_added)`.
#[allow(clippy::type_complexity)]
pub(crate) fn refine_or_extract(
    u: &mut Unroller<'_>,
    vals: &[u128],
) -> Result<CandidatePlan, usize> {
    // Pass 1: provisional nondet root cells, first-resolver-wins (a conflicting
    // later read shows up as a disagreement below and triggers congruence).
    let mut root_cells: HashMap<usize, HashMap<u128, u128>> = HashMap::new();
    let mut cell_source: HashMap<(usize, u128), usize> = HashMap::new();
    for (ri, r) in u.reads.iter().enumerate() {
        let (iw, _) = u.terms[r.term].dims();
        let idx = vals[r.idx_role] & mask(iw);
        if let Resolved::NondetCell { term, idx } = resolve_concrete(&u.terms, r.term, idx, vals) {
            let slot = root_cells.entry(term).or_default();
            if !slot.contains_key(&idx) {
                slot.insert(idx, vals[r.var_role]);
                cell_source.insert((term, idx), ri);
            }
        }
    }

    // Pass 2: check every read against its concrete chain value.
    let mut disagreeing: Vec<usize> = Vec::new();
    for (ri, r) in u.reads.iter().enumerate() {
        let (iw, ew) = u.terms[r.term].dims();
        let idx = vals[r.idx_role] & mask(iw);
        let true_val = match resolve_concrete(&u.terms, r.term, idx, vals) {
            Resolved::ByWrite(v) | Resolved::ByInitRoot(v) => v & mask(ew),
            Resolved::NondetCell { term, idx } => root_cells[&term][&idx] & mask(ew),
        };
        if vals[r.var_role] & mask(ew) != true_val {
            disagreeing.push(ri);
        }
    }
    if disagreeing.is_empty() {
        // Pass 3: array-equality consistency + completion (extensionality).
        return match eq_check_and_complete(u, vals, &mut root_cells) {
            Ok(root_defaults) => Ok(CandidatePlan {
                root_cells,
                root_defaults,
            }),
            Err(added) => Err(added),
        };
    }

    // Batch frontier instantiation (phase-3 F1+F2). For EVERY disagreeing
    // read, unfold its ENTIRE structural spine along the model resolution
    // path down to the root in one round, and — when it resolves to a nondet
    // root cell — unfold the SEEDING read's spine symmetrically, then add the
    // root Ackermann congruence pairs the new direct root reads enable.
    //
    // Completeness argument (the phase-2 one-link policy was provably
    // insufficient — both observed hard stalls had the seeder's first link
    // done but its deeper links never demanded): once every read on BOTH
    // model paths is axiom-done, each side's value is constraint-chained to a
    // direct root read, and the root congruence pair between the two sides
    // either already exists (then the SAT model would violate a hard
    // constraint — impossible) or is newly addable (progress). Hence
    // `added == 0` is unreachable for a read-table disagreement.
    //
    // Perf argument (the measured one-link ladder): depth d previously needed
    // ~2(d+1) refinement solves because each solve unfolded one link; the
    // batch unfold makes iterations-per-depth O(1), turning the quadratic
    // depth cost ~linear. Soundness: identical axiom vocabulary (ROW,
    // select-over-ite, const-root, root congruence — each a fact of array
    // semantics over the SSA epoch terms), just more instances per round; the
    // abstraction stays an over-approximation and the fresh-solver-per-query
    // discipline is unchanged. Termination: `axiom_done` flags are monotone
    // over a read table whose growth per unfold is one chain link, and chains
    // are finite at fixed depth.
    let mut added = 0usize;
    let mut root_targets: Vec<usize> = Vec::new();
    for &ri in &disagreeing {
        let (a, root_read) = unfold_model_path(u, ri, vals);
        added += a;
        root_targets.extend(root_read);
        // Symmetric walk for the seeding read of the contradicted cell.
        let (iw, _) = u.terms[u.reads[ri].term].dims();
        let idx = vals[u.reads[ri].idx_role] & mask(iw);
        if let Resolved::NondetCell { term, idx } =
            resolve_concrete(&u.terms, u.reads[ri].term, idx, vals)
        {
            if let Some(&src) = cell_source.get(&(term, idx)) {
                if src != ri {
                    let (a2, root_read2) = unfold_model_path(u, src, vals);
                    added += a2;
                    root_targets.extend(root_read2);
                }
            }
        }
    }
    for rr in root_targets {
        added += add_root_congruence_pairs(u, rr);
    }
    Err(added)
}

/// F1/F2 core: unfold the ENTIRE structural spine of read `ri` along its
/// model resolution path (the exact [`resolve_concrete`] walk) down to the
/// chain's resolution point, instantiating every link's structural axiom
/// (ROW / select-over-ite / const-root) and creating the lazy chain reads.
///
/// Returns `(axioms_added, Some(root_read))` when the path lands on a nondet
/// root — `root_read` is the direct root-level [`ReadEntry`] at `ri`'s index
/// node, the read that makes the root Ackermann congruence pair expressible —
/// and `(axioms_added, None)` when the path terminates at a model-equal write
/// or a const-init root (there the chain constraints alone pin the value).
///
/// Only `vals`-decodable roles are consulted (write index/value mirrors and
/// ite condition mirrors, all created at unroll time), never the roles of
/// reads created during this refinement round.
fn unfold_model_path(u: &mut Unroller<'_>, ri: usize, vals: &[u128]) -> (usize, Option<usize>) {
    let (iw, _) = u.terms[u.reads[ri].term].dims();
    let idx = vals[u.reads[ri].idx_role] & mask(iw);
    let mut added = 0usize;
    let mut cur = ri;
    loop {
        let term = u.terms[u.reads[cur].term];
        let idx_u = u.reads[cur].idx_u;
        match term {
            ATerm::Write {
                base, iw, idx_role, ..
            } => {
                added += usize::from(instantiate_structural(u, cur));
                if vals[idx_role] & mask(iw) == idx {
                    // Resolved by this write under the model: the ROW axiom
                    // just instantiated pins the value here.
                    return (added, None);
                }
                let Ok(nxt) = u.get_or_make_read(base, idx_u) else {
                    return (added, None);
                };
                cur = nxt;
            }
            ATerm::Ite {
                then_t,
                else_t,
                cond_role,
                ..
            } => {
                added += usize::from(instantiate_structural(u, cur));
                let branch = if vals[cond_role] & 1 == 1 {
                    then_t
                } else {
                    else_t
                };
                let Ok(nxt) = u.get_or_make_read(branch, idx_u) else {
                    return (added, None);
                };
                cur = nxt;
            }
            ATerm::RootInit { .. } => {
                added += usize::from(instantiate_structural(u, cur));
                return (added, None);
            }
            ATerm::RootNondet { .. } => return (added, Some(cur)),
        }
    }
}

/// Add every missing root Ackermann congruence pair between `target` (a read
/// directly on a nondet root term) and its peers on the same term:
/// `(idx_a == idx_b) => (val_a == val_b)`. Returns the number of new pairs.
pub(crate) fn add_root_congruence_pairs(u: &mut Unroller<'_>, target: usize) -> usize {
    let term = u.reads[target].term;
    let peers: Vec<usize> = (0..u.reads.len())
        .filter(|&pj| pj != target && u.reads[pj].term == term)
        .collect();
    let mut here = 0usize;
    for pj in peers {
        let key = (target.min(pj), target.max(pj));
        if !u.congruence_pairs.insert(key) {
            continue;
        }
        let (a, b) = (&u.reads[key.0], &u.reads[key.1]);
        let (ia, va, ib, vb) = (a.idx_u, a.var_u, b.idx_u, b.var_u);
        let eq_i = u.emit(Btor2Node::Eq, 1, vec![ia, ib]);
        let eq_v = u.emit(Btor2Node::Eq, 1, vec![va, vb]);
        let imp = u.emit(Btor2Node::Implies, 1, vec![eq_i, eq_v]);
        u.emit_constraint(imp);
        here += 1;
    }
    here
}

/// Determined value at index `j` of a model-concretized array side (`None` =
/// free nondet cell): outermost write override, else init default, else the
/// resolved/completed nondet root cell.
fn det_at(
    ov: &HashMap<u128, u128>,
    root: EffRoot,
    root_cells: &HashMap<usize, HashMap<u128, u128>>,
    em: u128,
    j: u128,
) -> Option<u128> {
    if let Some(&v) = ov.get(&j) {
        return Some(v & em);
    }
    match root {
        EffRoot::Init(d) => Some(d & em),
        EffRoot::Nondet(r) => root_cells.get(&r).and_then(|m| m.get(&j)).map(|&v| v & em),
    }
}

/// Pass 3 of `refine_or_extract`: check every UNEXPANDED eq entry's model
/// truth against the concrete/completable truth of its two chains under the
/// FIXED (exact, residual-domain) `word_eq` semantics, and either
///
/// * complete the assignment (mutating `root_cells`, returning the residual
///   default per nondet root) so every claimed `e = 1` replays as EQUAL and
///   every claimed `e = 0` is pinned genuinely unequal, or
/// * instantiate the missing axiom (E2 skolem / E3 index instance) and return
///   `Err(added)`.
///
/// E1-expanded entries are exact once the read table is chain-consistent
/// (which pass 2 just verified), so they are skipped. Completion conflicts
/// (two entries demanding different residual defaults for one root) become E3
/// instances at a fresh residual index — over iterations the conflict turns
/// into determined disagreeing cells the SAT solver must face; the per-depth
/// iteration/read caps bound the process (fail-closed decline on stall).
fn eq_check_and_complete(
    u: &mut Unroller<'_>,
    vals: &[u128],
    root_cells: &mut HashMap<usize, HashMap<u128, u128>>,
) -> Result<HashMap<usize, u128>, usize> {
    let mut root_defaults: HashMap<usize, u128> = HashMap::new();
    let mut added = 0usize;
    let mut all_consistent = true;

    for ei in 0..u.eqs.len() {
        if u.eqs[ei].expanded {
            continue;
        }
        let (term_a, term_b, iw, ew, var_role) = {
            let e = &u.eqs[ei];
            (e.term_a, e.term_b, e.iw, e.ew, e.var_role)
        };
        let e_model = vals[var_role] & 1 == 1;
        let em = mask(ew);
        let (root_a, ov_a) = effective_array(&u.terms, term_a, vals);
        let (root_b, ov_b) = effective_array(&u.terms, term_b, vals);

        // K = indices determined on at least one side.
        let mut k_set: HashSet<u128> = HashSet::new();
        k_set.extend(ov_a.keys().copied());
        k_set.extend(ov_b.keys().copied());
        for root in [root_a, root_b] {
            if let EffRoot::Nondet(r) = root {
                if let Some(m) = root_cells.get(&r) {
                    k_set.extend(m.keys().copied());
                }
            }
        }

        // First index determined on BOTH sides where the values disagree.
        let disagree_at = k_set.iter().copied().find(|&j| {
            matches!(
                (
                    det_at(&ov_a, root_a, root_cells, em, j),
                    det_at(&ov_b, root_b, root_cells, em, j),
                ),
                (Some(a), Some(b)) if a != b
            )
        });

        // Residual-domain emptiness, decided by arithmetic (mirrors word_eq).
        let residual_nonempty = if iw >= 64 {
            true
        } else {
            (k_set.len() as u128) < (1u128 << iw)
        };
        // Smallest index outside K (for residual-witness E3 instances).
        let fresh_residual = || -> u128 {
            let mut j = 0u128;
            while k_set.contains(&j) {
                j += 1;
            }
            j
        };

        // Definite inequality on the residual: both roots const-init with
        // differing defaults and a nonempty residual.
        let residual_definitely_unequal = residual_nonempty
            && matches!(
                (root_a, root_b),
                (EffRoot::Init(da), EffRoot::Init(db)) if da & em != db & em
            );

        if let Some(j) = disagree_at {
            if e_model {
                // Claimed equal, provably unequal at j: E3 there.
                all_consistent = false;
                added += usize::from(u.instantiate_e3(ei, j));
            }
            // Claimed unequal + genuinely unequal: consistent, no completion.
            continue;
        }
        if residual_definitely_unequal {
            if e_model {
                // Claimed equal, but the residual reads differing init
                // defaults: E3 at a residual witness index.
                all_consistent = false;
                added += usize::from(u.instantiate_e3(ei, fresh_residual()));
            }
            continue;
        }

        // From here: everything determined agrees, and the residual is
        // empty / equal-defaults / free — i.e. EQUAL or completable-equal.
        if !e_model {
            // Claimed unequal with no pinned disagreement: skolemize (E2) so
            // the solver must exhibit a concrete witness index, or flip `e`.
            all_consistent = false;
            added += usize::from(u.instantiate_skolem(ei));
            continue;
        }

        // Claimed equal: COMPLETE the free cells so the claim replays.
        // (a) K indices determined on exactly one side: mirror the value
        //     into the free side's nondet root.
        let mut conflict = false;
        for &j in &k_set {
            let da = det_at(&ov_a, root_a, root_cells, em, j);
            let db = det_at(&ov_b, root_b, root_cells, em, j);
            match (da, db) {
                (Some(_), Some(_)) => {}
                (Some(v), None) => {
                    if let (EffRoot::Nondet(r), false) = (root_b, ov_b.contains_key(&j)) {
                        root_cells.entry(r).or_default().insert(j, v);
                    } else {
                        conflict = true;
                    }
                }
                (None, Some(v)) => {
                    if let (EffRoot::Nondet(r), false) = (root_a, ov_a.contains_key(&j)) {
                        root_cells.entry(r).or_default().insert(j, v);
                    } else {
                        conflict = true;
                    }
                }
                (None, None) => {
                    // Defensive (every j in K is determined on >= 1 side by
                    // construction): pin both free cells to 0.
                    for (root, ov) in [(root_a, &ov_a), (root_b, &ov_b)] {
                        if let (EffRoot::Nondet(r), false) = (root, ov.contains_key(&j)) {
                            root_cells.entry(r).or_default().insert(j, 0);
                        }
                    }
                }
            }
        }
        // (b) Residual defaults: make both sides read the same value on the
        //     (infinite-ish) unwritten remainder.
        if residual_nonempty && !conflict {
            let require_default =
                |r: usize, d: u128, root_defaults: &mut HashMap<usize, u128>| -> bool {
                    match root_defaults.get(&r) {
                        None => {
                            root_defaults.insert(r, d & em);
                            true
                        }
                        Some(&prev) => prev == d & em,
                    }
                };
            match (root_a, root_b) {
                (EffRoot::Init(_), EffRoot::Init(_)) => {} // equal (checked above)
                (EffRoot::Nondet(ra), EffRoot::Nondet(rb)) => {
                    if ra != rb {
                        // A common residual value: reuse one already demanded
                        // of either root, else 0 (the extractor's natural
                        // default).
                        let d = root_defaults
                            .get(&ra)
                            .or_else(|| root_defaults.get(&rb))
                            .copied()
                            .unwrap_or(0);
                        conflict |= !require_default(ra, d, &mut root_defaults);
                        conflict |= !require_default(rb, d, &mut root_defaults);
                    }
                }
                (EffRoot::Nondet(r), EffRoot::Init(d)) | (EffRoot::Init(d), EffRoot::Nondet(r)) => {
                    conflict |= !require_default(r, d, &mut root_defaults);
                }
            }
        }
        if conflict {
            // Completion conflict: force the disagreement into the formula at
            // a fresh residual index; over iterations it becomes a determined
            // disagreement the solver must resolve.
            all_consistent = false;
            added += usize::from(u.instantiate_e3(ei, fresh_residual()));
        }
    }

    if all_consistent {
        Ok(root_defaults)
    } else {
        Err(added)
    }
}

/// Instantiate the structural axiom for read `ri` (one chain link: ROW,
/// select-over-ite, or const-root). Returns `true` iff a new constraint was
/// added. No-op (false) when the axiom is already present.
pub(crate) fn instantiate_structural(u: &mut Unroller<'_>, ri: usize) -> bool {
    if u.reads[ri].axiom_done {
        return false;
    }
    let idx_u = u.reads[ri].idx_u;
    let var_u = u.reads[ri].var_u;
    match u.terms[u.reads[ri].term] {
        ATerm::Write {
            base,
            ew,
            idx_u: widx_u,
            val_u: wval_u,
            ..
        } => {
            // ROW: r == ite(idx == wi, wv, read(base, idx)).
            let Ok(base_read) = u.get_or_make_read(base, idx_u) else {
                return false;
            };
            let base_var = u.reads[base_read].var_u;
            let eq_i = u.emit(Btor2Node::Eq, 1, vec![idx_u, widx_u]);
            let ite = u.emit(Btor2Node::Ite, ew, vec![eq_i, wval_u, base_var]);
            let ax = u.emit(Btor2Node::Eq, 1, vec![var_u, ite]);
            u.emit_constraint(ax);
        }
        ATerm::Ite {
            then_t,
            else_t,
            ew,
            cond_u,
            ..
        } => {
            // Select-over-ite: r == ite(c, read(a, idx), read(b, idx)).
            let (Ok(rt), Ok(re)) = (
                u.get_or_make_read(then_t, idx_u),
                u.get_or_make_read(else_t, idx_u),
            ) else {
                return false;
            };
            let vt = u.reads[rt].var_u;
            let ve = u.reads[re].var_u;
            let ite = u.emit(Btor2Node::Ite, ew, vec![cond_u, vt, ve]);
            let ax = u.emit(Btor2Node::Eq, 1, vec![var_u, ite]);
            u.emit_constraint(ax);
        }
        ATerm::RootInit { ew, default, .. } => {
            // Const-array root: r == default.
            let d = u.emit(Btor2Node::ConstD(default.to_string()), ew, vec![]);
            let ax = u.emit(Btor2Node::Eq, 1, vec![var_u, d]);
            u.emit_constraint(ax);
        }
        ATerm::RootNondet { .. } => return false,
    }
    u.reads[ri].axiom_done = true;
    true
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

/// Outcome of closing one depth's query (solve + refine to fixpoint).
enum DepthOutcome {
    /// The depth's query is UNSAT on the over-approximation ⇒ genuinely no
    /// counterexample of this shape at this depth.
    Closed,
    /// A chain-consistent, replay-VALIDATED concrete counterexample.
    Unsafe {
        depth: usize,
        fired: Vec<usize>,
        model: WordLevelModel,
        witness: Option<Btor2Witness>,
    },
    /// A chain-consistent abstract model in a mode that must not extract a
    /// trace (k-induction step: frame 0 is free, so the model is not a real
    /// trace) — the query is satisfiable, nothing more.
    AbstractCex,
}

/// Per-depth resource caps (from the lane configs).
struct DepthCaps {
    max_refinements: usize,
    verbose: bool,
}

/// F3: the derived per-depth read cap (replaces the flat `max_reads` wall the
/// profiling run showed declining ifcomp at depth 11 with a healthy ~50
/// reads/depth growth).
///
/// `R1` is the measured per-net read footprint — the read-table size after
/// the first depth >= 1 closes (two frames' worth of reads, the net's own
/// scale; nothing is assumed about it). The cap then grows linearly:
///
/// ```text
/// cap(t) = max(config_floor, SLACK * R1 * (t + 1))
/// ```
///
/// with ONE global slack multiplier covering the batch-unfold worst case
/// (both sides of every disagreement unfolded to the root roughly doubles
/// the resolution-path reads, plus their lazily created chain peers). Until
/// `R1` is measured the flat config floor applies unchanged. The cap is
/// structurally bounded by `SLACK * R1 * (max_depth + 1)` — a per-net derived
/// ceiling, not a magic constant. Exceeding the cap only ever produces
/// [`ArrayBmcOutcome::Declined`] — fail-closed, no soundness surface.
struct ReadCap {
    floor: usize,
    per_frame: Option<usize>,
}

impl ReadCap {
    /// The single global slack multiplier (see the struct docs).
    const SLACK: usize = 4;

    fn new(floor: usize) -> Self {
        ReadCap {
            floor,
            per_frame: None,
        }
    }

    /// Record the measured footprint after a depth closes (first depth >= 1
    /// wins; later depths carry refinement growth, not the base footprint).
    fn observe(&mut self, depth_closed: usize, reads_now: usize) {
        if depth_closed >= 1 && self.per_frame.is_none() {
            self.per_frame = Some(reads_now.max(1));
        }
    }

    fn cap(&self, t: usize) -> usize {
        match self.per_frame {
            None => self.floor,
            Some(r1) => self
                .floor
                .max(r1.saturating_mul(Self::SLACK).saturating_mul(t + 1)),
        }
    }
}

/// Close the depth-`t` query on `u`: solve fresh, refine spurious models via
/// lazy axiom instantiation, and loop to a definitive outcome or a
/// fail-closed error. When `extract_cex` is false (k-induction step mode), a
/// chain-consistent SAT returns [`DepthOutcome::AbstractCex`] instead of
/// extracting and replaying a model.
fn close_depth(
    program: &Btor2Program,
    u: &mut Unroller<'_>,
    t: usize,
    caps: &DepthCaps,
    read_cap: &ReadCap,
    over_budget: &dyn Fn(&str) -> Option<String>,
    extract_cex: bool,
) -> Result<DepthOutcome, String> {
    let mut iterations = 0usize;
    loop {
        if let Some(r) = over_budget("refinement loop") {
            return Err(r);
        }
        if iterations >= caps.max_refinements {
            return Err(format!(
                "refinement iteration cap ({}) hit at depth {t}",
                caps.max_refinements
            ));
        }
        let cap = read_cap.cap(t);
        if u.reads.len() > cap {
            return Err(format!("read-table cap ({cap}) exceeded at depth {t}"));
        }
        iterations += 1;

        let query = u.build_query(t)?;
        let circuit = bitblast(&query, 128).map_err(|e| format!("bit-blast: {e}"))?;
        match solve_query(&circuit)? {
            QueryResult::Unsat => {
                // Over-approximation UNSAT ⇒ genuinely no cex at depth t.
                // A bounded fact only — never surfaced as a verdict.
                if caps.verbose {
                    eprintln!(
                        "array-bmc: depth {t} closed (iter {iterations}, {} reads, {} axiom constraints)",
                        u.reads.len(),
                        u.constraints_ids.len()
                    );
                }
                return Ok(DepthOutcome::Closed);
            }
            QueryResult::Sat(model) => {
                let vals = decode_input_values(&circuit, &model);
                if vals.len() != u.input_roles.len() {
                    return Err(format!(
                        "input decode arity mismatch: {} bits tables vs {} roles",
                        vals.len(),
                        u.input_roles.len()
                    ));
                }
                match refine_or_extract(u, &vals) {
                    Err(added) => {
                        if added == 0 {
                            // No progress possible — fail closed rather
                            // than loop or guess.
                            return Err(format!(
                                "refinement stalled at depth {t} (spurious model, no new axiom)"
                            ));
                        }
                        if caps.verbose {
                            eprintln!(
                                "array-bmc: depth {t} iter {iterations}: spurious model, +{added} axiom(s)"
                            );
                        }
                        continue;
                    }
                    Ok(plan) => {
                        if !extract_cex {
                            return Ok(DepthOutcome::AbstractCex);
                        }
                        // Chain-consistent: build the candidate model and
                        // gate it on concrete forward replay.
                        let model = extract_model(program, u, &vals, &plan, t);
                        let Some(fired) =
                            replay_collect_bad(program, &model.initial, &model.input_frames)
                        else {
                            // A chain-consistent model that does not
                            // replay means a semantic mismatch between
                            // the unrolling and the replayer — never
                            // report, decline loudly.
                            return Err(format!(
                                "chain-consistent candidate at depth {t} failed concrete \
                                 replay (cross-lane semantic mismatch) — declined, not reported"
                            ));
                        };
                        let witness = build_word_level_witness(program, &model);
                        return Ok(DepthOutcome::Unsafe {
                            depth: t,
                            fired,
                            model,
                            witness,
                        });
                    }
                }
            }
        }
    }
}

fn check_inner(program: &Btor2Program, config: &ArrayBmcConfig) -> Result<ArrayBmcOutcome, String> {
    lane_supported(program)?;
    let start = Instant::now();
    let over_budget = |what: &str| -> Option<String> {
        config
            .time_budget
            .and_then(|b| (start.elapsed() > b).then(|| format!("time budget exceeded ({what})")))
    };
    let caps = DepthCaps {
        max_refinements: config.max_refinements_per_depth,
        verbose: config.verbose,
    };
    let mut read_cap = ReadCap::new(config.max_reads);

    let mut u = Unroller::new(program, config.max_reads, false)?;
    let mut depth_reached = None;

    for t in 0..=config.max_depth {
        if let Some(r) = over_budget("depth loop") {
            return Err(r);
        }
        match close_depth(program, &mut u, t, &caps, &read_cap, &over_budget, true)? {
            DepthOutcome::Closed => {
                depth_reached = Some(t);
                read_cap.observe(t, u.reads.len());
            }
            DepthOutcome::Unsafe {
                depth,
                fired,
                model,
                witness,
            } => {
                return Ok(ArrayBmcOutcome::Unsafe {
                    depth,
                    fired,
                    model,
                    witness,
                })
            }
            DepthOutcome::AbstractCex => {
                return Err("internal: abstract cex in extraction mode".into())
            }
        }
    }

    Ok(ArrayBmcOutcome::BoundedNoCex {
        depth_reached: depth_reached.unwrap_or(0),
    })
}

// ---------------------------------------------------------------------------
// K-induction over the lazy core (the phase-2 stepping stone toward IC3)
// ---------------------------------------------------------------------------

/// Configuration for the k-induction lane.
#[derive(Debug, Clone)]
pub struct ArrayKindConfig {
    /// Largest induction depth to try (k = 1..=max_k).
    pub max_k: usize,
    /// Maximum refinement iterations per query (axiom-instantiation rounds).
    pub max_refinements_per_depth: usize,
    /// Maximum read-table entries per unroller.
    pub max_reads: usize,
    /// Wall-clock budget for the whole lane; `None` = unbounded.
    pub time_budget: Option<Duration>,
    /// Print progress to stderr.
    pub verbose: bool,
}

impl Default for ArrayKindConfig {
    fn default() -> Self {
        ArrayKindConfig {
            max_k: 10,
            max_refinements_per_depth: 64,
            max_reads: 512,
            time_budget: None,
            verbose: false,
        }
    }
}

/// Outcome of the k-induction lane.
#[derive(Debug)]
pub enum ArrayKindOutcome {
    /// UNBOUNDED SAFE: every property is k-inductive, with BOTH the base
    /// completion (depths `0..k` closed) and the step query INDEPENDENTLY
    /// re-discharged through the LRAT-checked second trust path
    /// ([`crate::array_cert`]'s disjoint leaf). This is the ONLY variant that
    /// is an unbounded-safe verdict, and it is minted nowhere else.
    ProvedSafe {
        /// The induction depth that closed.
        k: usize,
    },
    /// The base BMC found a concrete counterexample (replay-validated, same
    /// gate as [`ArrayBmcOutcome::Unsafe`]).
    Unsafe {
        /// Frame index at which the bad property fires.
        depth: usize,
        /// Indices into `program.bad_properties` that fire (from replay).
        fired: Vec<usize>,
        /// The replay-validated concrete model.
        model: WordLevelModel,
        /// btorsim witness when serializable (fail-closed `None` otherwise).
        witness: Option<Btor2Witness>,
    },
    /// No counterexample within `depth_reached` transitions and no k became
    /// inductive (or the independent discharge did not verify — the claim is
    /// DOWNGRADED here, never reported unbounded). A bounded fact only.
    BoundedNoCex {
        /// The last base depth that was closed.
        depth_reached: usize,
    },
    /// Outside the supported slice or a budget/iteration cap. The caller's
    /// default decision tree proceeds unchanged.
    Declined {
        /// Why the lane declined.
        reason: String,
    },
}

/// K-induction over the lazy-array core. Base case = the per-depth BMC loop
/// (real `init`, depths `0..k` closed). Step case = ONE extra query shape on
/// unchanged machinery: frame 0 fully nondeterministic (`init` ignored),
/// `constraint`s assumed at every frame, NOT-bad assumed at frames
/// `0..k-1`, bad asserted at frame `k`. Abstraction-UNSAT ⇒ concrete
/// step-UNSAT (the over-approximation direction is the sound one) ⇒ with the
/// closed base, unbounded safe — but [`ArrayKindOutcome::ProvedSafe`] is
/// reported ONLY after every one of those queries is re-discharged through
/// the independent LRAT-checked leaf; otherwise the claim downgrades to
/// [`ArrayKindOutcome::BoundedNoCex`]. No simple-path constraints yet
/// (completeness-only; soundness unaffected).
#[must_use]
pub fn check_array_kinduction(
    program: &Btor2Program,
    config: &ArrayKindConfig,
) -> ArrayKindOutcome {
    match kind_inner(program, config) {
        Ok(outcome) => outcome,
        Err(reason) => ArrayKindOutcome::Declined { reason },
    }
}

fn kind_inner(
    program: &Btor2Program,
    config: &ArrayKindConfig,
) -> Result<ArrayKindOutcome, String> {
    lane_supported(program)?;
    let start = Instant::now();
    let over_budget = |what: &str| -> Option<String> {
        config
            .time_budget
            .and_then(|b| (start.elapsed() > b).then(|| format!("time budget exceeded ({what})")))
    };
    let caps = DepthCaps {
        max_refinements: config.max_refinements_per_depth,
        verbose: config.verbose,
    };
    // One shared derived cap: R1 measured on the base unroller (same net,
    // same per-frame read footprint) governs both unrollers.
    let mut read_cap = ReadCap::new(config.max_reads);

    // Base unroller: the ordinary BMC lane (real init). Step unroller: frame
    // 0 fully free. Both persist across k so axiom instances accumulate.
    let mut u_base = Unroller::new(program, config.max_reads, false)?;
    let mut u_step = Unroller::new(program, config.max_reads, true)?;
    // Frames `0..notbad_emitted` carry the NOT-bad step assumptions.
    let mut notbad_emitted = 0usize;
    let mut base_closed: Option<usize> = None;

    for k in 1..=config.max_k {
        if let Some(r) = over_budget("k loop") {
            return Err(r);
        }

        // Base: close depth k-1 with real init (replay-gated cex if SAT).
        match close_depth(
            program,
            &mut u_base,
            k - 1,
            &caps,
            &read_cap,
            &over_budget,
            true,
        )? {
            DepthOutcome::Closed => {
                base_closed = Some(k - 1);
                read_cap.observe(k - 1, u_base.reads.len());
            }
            DepthOutcome::Unsafe {
                depth,
                fired,
                model,
                witness,
            } => {
                return Ok(ArrayKindOutcome::Unsafe {
                    depth,
                    fired,
                    model,
                    witness,
                })
            }
            DepthOutcome::AbstractCex => {
                return Err("internal: abstract cex in base (extraction) mode".into())
            }
        }

        // Step: assume NOT-bad at frames 0..k-1 (accumulates across k).
        while notbad_emitted < k {
            let f = notbad_emitted;
            u_step.seed_through(f)?;
            for i in 0..program.bad_properties.len() {
                let bid = program.bad_properties[i];
                let cond = u_step.node_cond(bid)?;
                let cu = u_step.unroll_ref(f, cond)?;
                let ncu = u_step.emit(Btor2Node::Not, 1, vec![cu]);
                u_step.emit_constraint(ncu);
            }
            notbad_emitted += 1;
        }

        // Step query: bad at frame k under the NOT-bad prefix, free frame 0.
        match close_depth(
            program,
            &mut u_step,
            k,
            &caps,
            &read_cap,
            &over_budget,
            false,
        )? {
            DepthOutcome::AbstractCex => {
                // Not k-inductive at this k (the abstract model may or may
                // not be concretely real — either way, try a deeper k).
                if config.verbose {
                    eprintln!("array-kind: step SAT at k={k} — not k-inductive, trying k+1");
                }
            }
            DepthOutcome::Closed => {
                // k-inductive on the abstraction. UNBOUNDED SAFE is minted
                // ONLY if the base completion AND the step re-discharge
                // through the independent LRAT-checked path.
                if config.verbose {
                    eprintln!("array-kind: step closed at k={k} — running independent discharge");
                }
                match discharge_kind_independent(&mut u_base, &mut u_step, k) {
                    Ok(()) => return Ok(ArrayKindOutcome::ProvedSafe { k }),
                    Err(why) => {
                        // DOWNGRADE (never report unbounded): the bounded
                        // base facts stand on the primary loop's evidence.
                        if config.verbose {
                            eprintln!(
                                "array-kind: independent discharge FAILED ({why}) — \
                                 downgrading to bounded-no-cex"
                            );
                        }
                        return Ok(ArrayKindOutcome::BoundedNoCex {
                            depth_reached: k - 1,
                        });
                    }
                }
            }
            DepthOutcome::Unsafe { .. } => {
                return Err("internal: extraction in step (abstract) mode".into())
            }
        }
    }

    Ok(ArrayKindOutcome::BoundedNoCex {
        depth_reached: base_closed.unwrap_or(0),
    })
}

/// The independent second trust path for a k-induction claim: re-discharge
/// the base queries (depths `0..k`, real init — with the FINAL axiom set,
/// which only strengthens soundly: axioms are facts of array semantics) and
/// the step query (frame k), each as a bit-blasted CNF whose UNSAT must
/// carry an LRAT proof verified by the separate `ay-lrat-check` crate. Any
/// failure is an error — the caller downgrades, never reports.
fn discharge_kind_independent(
    u_base: &mut Unroller<'_>,
    u_step: &mut Unroller<'_>,
    k: usize,
) -> Result<(), String> {
    for t in 0..k {
        let query = u_base.build_query(t)?;
        let circuit = bitblast(&query, 128).map_err(|e| format!("bit-blast(base {t}): {e}"))?;
        discharge_unsat_lrat(&circuit).map_err(|e| format!("base depth {t}: {e}"))?;
    }
    let query = u_step.build_query(k)?;
    let circuit = bitblast(&query, 128).map_err(|e| format!("bit-blast(step): {e}"))?;
    discharge_unsat_lrat(&circuit).map_err(|e| format!("step at k={k}: {e}"))?;
    Ok(())
}

/// Build the candidate [`WordLevelModel`] from the decoded input values:
/// frame-0 nondet scalars and per-frame inputs directly from their roles;
/// frame-0 nondet arrays from the resolved root cells plus the
/// extensionality completion (per-root residual defaults — default 0 when no
/// eq entry constrains the residual; a nonzero completed default replays
/// exactly but cannot serialize to a btorsim witness, which fails closed to
/// `witness: None` while the verdict stays replay-proven).
fn extract_model(
    program: &Btor2Program,
    u: &Unroller<'_>,
    vals: &[u128],
    plan: &CandidatePlan,
    depth: usize,
) -> WordLevelModel {
    let mut initial = InitialState::default();
    let mut input_frames: Vec<InputFrame> = vec![InputFrame::default(); depth + 1];

    for (i, role) in u.input_roles.iter().enumerate() {
        match role {
            InputRole::Frame0State { src } => {
                let width = u
                    .line_index
                    .get(src)
                    .and_then(|l| program.sorts.get(&l.sort_id))
                    .and_then(|s| match s {
                        Btor2Sort::BitVec(w) => Some(*w),
                        Btor2Sort::Array { .. } => None,
                    })
                    .unwrap_or(128);
                initial.states.insert(
                    *src,
                    WordValue::Bv {
                        bits: vals[i] & mask(width),
                        width,
                    },
                );
            }
            InputRole::FrameInput { frame, src } => {
                if *frame <= depth {
                    input_frames[*frame].insert(*src, vals[i]);
                }
            }
            InputRole::ReadVar
            | InputRole::Mirror
            | InputRole::EqVar
            | InputRole::EqSkolem
            | InputRole::FreeIndex => {}
        }
    }

    // Nondet array roots -> frame-0 array values (resolved cells + completed
    // residual default).
    for (ti, term) in u.terms.iter().enumerate() {
        if let ATerm::RootNondet { state_id, iw, ew } = term {
            let cells = plan
                .root_cells
                .get(&ti)
                .map(|m| m.iter().map(|(&k, &v)| (k, v & mask(*ew))).collect())
                .unwrap_or_default();
            let default = plan.root_defaults.get(&ti).copied().unwrap_or(0) & mask(*ew);
            initial.states.insert(
                *state_id,
                WordValue::Array {
                    index_width: *iw,
                    elem_width: *ew,
                    default,
                    cells,
                },
            );
        }
    }

    WordLevelModel {
        num_frames: depth + 1,
        initial,
        input_frames,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn run(net: &str) -> ArrayBmcOutcome {
        let prog = parse(net).expect("parse");
        check_array_bmc(&prog, &ArrayBmcConfig::default())
    }

    fn run_depth(net: &str, k: usize) -> ArrayBmcOutcome {
        let prog = parse(net).expect("parse");
        check_array_bmc(
            &prog,
            &ArrayBmcConfig {
                max_depth: k,
                ..ArrayBmcConfig::default()
            },
        )
    }

    /// WIDE-index net (bv16 index — bit-blast-INELIGIBLE, the lane's target
    /// class): mem init 0, next writes 5 at index 0, bad = (mem[0] == 5).
    /// UNSAFE at depth 1, replay-proven; the witness serializes.
    #[test]
    fn wide_index_write_then_read_unsafe() {
        let net = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
8 zero 1
10 constd 2 5
13 write 3 4 8 10
7 next 3 4 13
9 read 2 4 8
12 sort bitvec 1
11 eq 12 9 10
14 bad 11
";
        // Confirm this really is outside the bit-blast lane.
        let prog = parse(net).expect("parse");
        assert!(
            crate::bitblast_eligible(&prog, 32).is_err(),
            "fixture must be bit-blast-ineligible (iw=16)"
        );
        match run(net) {
            ArrayBmcOutcome::Unsafe {
                depth,
                fired,
                witness,
                ..
            } => {
                assert_eq!(depth, 1);
                assert_eq!(fired, vec![0]);
                let w = witness.expect("zero-default init array must serialize");
                assert!(w.to_btor2_string().starts_with("sat\nb0\n"));
            }
            other => panic!("expected Unsafe, got {other:?}"),
        }
    }

    /// Phase-3 F1/F2 regression: the two-chains-one-cell hard-stall shape
    /// observed on the real ifcomp/ifcompf nets. A nondet wide root read
    /// through TWO different write chains at the same untouched cell (all
    /// write indices constrained away from the probe index) — the property
    /// (functional consistency: both reads equal) is unconditionally true,
    /// but the phase-2 one-link refinement policy could reach `added == 0`
    /// here: the longer chain's read SEEDS the root cell (agrees with itself,
    /// `axiom_done` after one link), the shorter chain unfolds to a direct
    /// root read with no root-level peer, and the one-link cell-source
    /// fallback no-ops. The batch frontier unfold must close every depth
    /// (never a "no new axiom" decline) and k-induction must prove it.
    const TWO_CHAINS_ONE_CELL: &str = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 mem
5 input 1 i1
6 input 1 i2
7 input 1 i3
8 input 2 v1
9 input 2 v2
10 input 2 v3
11 write 3 4 5 8
12 write 3 11 6 9
13 write 3 4 7 10
14 constd 1 7
15 read 2 12 14
16 read 2 13 14
17 sort bitvec 1
18 neq 17 5 14
19 constraint 18
20 neq 17 6 14
21 constraint 20
22 neq 17 7 14
23 constraint 22
24 neq 17 15 16
25 bad 24
26 next 3 4 4
";

    #[test]
    fn two_chains_one_cell_stall_shape_closes() {
        let prog = parse(TWO_CHAINS_ONE_CELL).expect("parse");
        assert!(
            crate::bitblast_eligible(&prog, 32).is_err(),
            "fixture must be bit-blast-ineligible (iw=16)"
        );
        match run_depth(TWO_CHAINS_ONE_CELL, 3) {
            ArrayBmcOutcome::BoundedNoCex { depth_reached } => assert_eq!(depth_reached, 3),
            other => panic!("expected BoundedNoCex (stall shape must close), got {other:?}"),
        }
    }

    #[test]
    fn two_chains_one_cell_kinduction_proves() {
        let prog = parse(TWO_CHAINS_ONE_CELL).expect("parse");
        let outcome = check_array_kinduction(&prog, &ArrayKindConfig::default());
        match outcome {
            ArrayKindOutcome::ProvedSafe { .. } => {}
            other => panic!(
                "expected ProvedSafe (step must refine to UNSAT, never stall), got {other:?}"
            ),
        }
    }

    /// SAFE twin (bad looks for 6, the chain writes 5): every depth closes
    /// UNSAT — reported ONLY as a bounded no-cex, never a verdict.
    #[test]
    fn wide_index_safe_is_bounded_only() {
        let net = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
8 zero 1
10 constd 2 5
13 write 3 4 8 10
7 next 3 4 13
9 read 2 4 8
12 sort bitvec 1
15 constd 2 6
11 eq 12 9 15
14 bad 11
";
        match run_depth(net, 4) {
            ArrayBmcOutcome::BoundedNoCex { depth_reached } => assert_eq!(depth_reached, 4),
            other => panic!("expected BoundedNoCex, got {other:?}"),
        }
    }

    /// Nondeterministic wide array: no init, bad = (mem[3] == 7) at frame 0.
    /// The lane must produce a frame-0 array model (cells from read
    /// valuations) that replays.
    #[test]
    fn nondet_wide_array_unsafe_at_frame0() {
        let net = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 mem
5 next 3 4 4
6 constd 1 3
7 read 2 4 6
8 constd 2 7
9 sort bitvec 1
10 eq 9 7 8
11 bad 10
";
        match run(net) {
            ArrayBmcOutcome::Unsafe { depth, model, .. } => {
                assert_eq!(depth, 0);
                match model.initial.states.get(&4).expect("initial mem") {
                    WordValue::Array { cells, .. } => {
                        assert_eq!(cells.get(&3).copied(), Some(7), "mem[3] must be 7");
                    }
                    other => panic!("expected array model, got {other:?}"),
                }
            }
            other => panic!("expected Unsafe, got {other:?}"),
        }
    }

    /// Aliasing/congruence with a symbolic index: two reads on the same nondet
    /// root must agree when their indices coincide. bad = (mem[i] != mem[j])
    /// AND (i == j) is UNSAT at every depth — only the congruence axiom class
    /// closes it.
    #[test]
    fn congruence_closes_aliasing() {
        let net = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 mem
5 next 3 4 4
6 input 1 i
7 input 1 j
8 read 2 4 6
9 read 2 4 7
10 sort bitvec 1
11 neq 10 8 9
12 eq 10 6 7
13 and 10 11 12
14 bad 13
";
        match run_depth(net, 2) {
            ArrayBmcOutcome::BoundedNoCex { .. } => {}
            other => panic!("expected BoundedNoCex (congruence), got {other:?}"),
        }
    }

    /// Read-over-write with a symbolic write index: read(write(mem,wi,wv), wi)
    /// must equal wv (McCarthy across the epoch chain), so bad = (read != wv)
    /// closes at every depth.
    #[test]
    fn row_axiom_closes_mccarthy() {
        let net = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 mem
5 input 1 wi
6 input 2 wv
7 write 3 4 5 6
8 next 3 4 7
9 read 2 7 5
10 sort bitvec 1
11 neq 10 9 6
12 bad 11
";
        match run_depth(net, 2) {
            ArrayBmcOutcome::BoundedNoCex { .. } => {}
            other => panic!("expected BoundedNoCex (ROW), got {other:?}"),
        }
    }

    /// Across-step epoch soundness: the value written at step 0 is observable
    /// at frame 1 but OVERWRITTEN at step 1 — a time-blind (single-epoch)
    /// abstraction would conflate the epochs. bad = (mem[0] == 1) AND
    /// (prev == 2) where prev latches mem[0]: fires exactly when mem[0]
    /// transitions 2 -> 1 across a step.
    #[test]
    fn per_epoch_consistency_across_steps() {
        let net = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
7 input 2 v
8 zero 1
9 write 3 4 8 7
10 next 3 4 9
11 state 2 prev
12 init 2 11 5
13 read 2 4 8
14 next 2 11 13
15 sort bitvec 1
16 constd 2 1
17 constd 2 2
18 eq 15 13 16
19 eq 15 11 17
20 and 15 18 19
21 bad 20
";
        // Frame f: prev = mem[0]@(f-1), mem[0] = v@(f-1). Need prev==2 and
        // mem[0]==1: v=2 at step 0, v=1 at step 1, bad at frame 2.
        match run(net) {
            ArrayBmcOutcome::Unsafe { depth, .. } => assert_eq!(depth, 2),
            other => panic!("expected Unsafe at depth 2, got {other:?}"),
        }
    }

    /// Ite-of-array distribution: reading through ite(c, write(mem,0,1), mem)
    /// must see 1 iff c. bad = (read != 1) AND c is closed; bad = (read == 1)
    /// AND c fires.
    #[test]
    fn select_over_ite_distributes() {
        let safe = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
7 sort bitvec 1
8 input 7 c
9 zero 1
10 constd 2 1
11 write 3 4 9 10
12 ite 3 8 11 4
13 next 3 4 12
14 read 2 12 9
15 neq 7 14 10
16 and 7 15 8
17 bad 16
";
        match run_depth(safe, 2) {
            ArrayBmcOutcome::BoundedNoCex { .. } => {}
            other => panic!("expected BoundedNoCex, got {other:?}"),
        }
        let unsafe_net = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
7 sort bitvec 1
8 input 7 c
9 zero 1
10 constd 2 1
11 write 3 4 9 10
12 ite 3 8 11 4
13 next 3 4 12
14 read 2 12 9
15 eq 7 14 10
16 and 7 15 8
17 bad 16
";
        match run(unsafe_net) {
            ArrayBmcOutcome::Unsafe { depth, .. } => assert_eq!(depth, 0),
            other => panic!("expected Unsafe at depth 0, got {other:?}"),
        }
    }

    /// Constraints participate at every frame (assume semantics): the input
    /// that would fire bad is excluded by a constraint.
    #[test]
    fn constraints_prune_bad() {
        let net = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
7 input 2 v
8 zero 1
9 write 3 4 8 7
10 next 3 4 9
11 read 2 4 8
12 sort bitvec 1
13 constd 2 9
14 eq 12 11 13
15 bad 14
16 neq 12 7 13
17 constraint 16
";
        match run_depth(net, 3) {
            ArrayBmcOutcome::BoundedNoCex { .. } => {}
            other => panic!("expected BoundedNoCex (constraint), got {other:?}"),
        }
    }

    // -- extensionality (phase 2) ------------------------------------------------

    /// Phase-1 declined this exact net ("array equality — declined"); phase 2
    /// handles it: two const-0-init wide arrays are extensionally EQUAL, so
    /// bad = eq(a, b) fires at frame 0 (replay-gated through word_eq).
    #[test]
    fn wide_eq_of_identical_const_arrays_fires() {
        let net = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 a
5 state 3 b
6 zero 2
7 init 3 4 6
8 init 3 5 6
9 next 3 4 4
10 next 3 5 5
11 sort bitvec 1
12 eq 11 4 5
13 bad 12
";
        match run_depth(net, 2) {
            ArrayBmcOutcome::Unsafe { depth, fired, .. } => {
                assert_eq!(depth, 0);
                assert_eq!(fired, vec![0]);
            }
            other => panic!("expected Unsafe@0, got {other:?}"),
        }
    }

    /// Differing-defaults twin: a init 0, b init 1 (wide domain, sparse
    /// writes) — extensionally UNEQUAL at every frame, so bad = eq(a, b)
    /// closes at every depth (E3 residual-witness instances), and the neq
    /// twin fires at frame 0 (E2 skolem pinning a witness index would also
    /// suffice, but the residual rule already decides it).
    #[test]
    fn wide_eq_differing_defaults_is_bounded_safe_and_neq_fires() {
        let eq_net = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 a
5 state 3 b
6 zero 2
7 init 3 4 6
8 one 2
9 init 3 5 8
10 next 3 4 4
11 next 3 5 5
12 sort bitvec 1
13 eq 12 4 5
14 bad 13
";
        match run_depth(eq_net, 2) {
            ArrayBmcOutcome::BoundedNoCex { depth_reached } => assert_eq!(depth_reached, 2),
            other => panic!("expected BoundedNoCex, got {other:?}"),
        }
        let neq_net = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 a
5 state 3 b
6 zero 2
7 init 3 4 6
8 one 2
9 init 3 5 8
10 next 3 4 4
11 next 3 5 5
12 sort bitvec 1
13 neq 12 4 5
14 bad 13
";
        match run_depth(neq_net, 2) {
            ArrayBmcOutcome::Unsafe { depth, .. } => assert_eq!(depth, 0),
            other => panic!("expected Unsafe@0, got {other:?}"),
        }
    }

    /// Completion path (E2/E3 + residual completion): a is NONDET (no init),
    /// b is const-0-init; bad = eq(a, b) must fire at frame 0 — the model
    /// claims e=1 and the extractor completes a's unread residual to b's
    /// default so the claim replays through the exact word_eq.
    #[test]
    fn wide_eq_nondet_vs_const_completes_and_fires() {
        let net = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 a
5 state 3 b
6 zero 2
7 init 3 5 6
8 next 3 4 4
9 next 3 5 5
10 sort bitvec 1
11 eq 10 4 5
12 bad 11
";
        match run_depth(net, 1) {
            ArrayBmcOutcome::Unsafe { depth, model, .. } => {
                assert_eq!(depth, 0);
                // The extracted nondet array must be extensionally all-0
                // (completed default 0, any explicit cells equal 0).
                match model.initial.states.get(&4).expect("initial a") {
                    WordValue::Array { default, cells, .. } => {
                        assert_eq!(*default, 0);
                        assert!(cells.values().all(|&v| v == 0), "cells: {cells:?}");
                    }
                    other => panic!("expected array, got {other:?}"),
                }
            }
            other => panic!("expected Unsafe@0, got {other:?}"),
        }
    }

    /// E3 equal-side propagation: eq(a, b) AND read(a, 5) == 7 with b
    /// const-0-init is UNSAT at every depth — the lazy eq abstraction alone
    /// would claim it, and only the E3 instance at index 5 (plus the read
    /// axioms) refutes it.
    #[test]
    fn wide_eq_e3_refutes_contradictory_read() {
        let net = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 a
5 state 3 b
6 zero 2
7 init 3 5 6
8 next 3 4 4
9 next 3 5 5
10 sort bitvec 1
11 eq 10 4 5
12 constd 1 5
13 read 2 4 12
14 constd 2 7
15 eq 10 13 14
16 and 10 11 15
17 bad 16
";
        match run_depth(net, 2) {
            ArrayBmcOutcome::BoundedNoCex { depth_reached } => assert_eq!(depth_reached, 2),
            other => panic!("expected BoundedNoCex (E3), got {other:?}"),
        }
    }

    /// E2 skolem: neq(a, b) with BOTH arrays const-0-init is UNSAT at every
    /// depth — the skolem read pair collapses to the shared default, so
    /// `!e => a[k] != b[k]` refutes every e=0 claim.
    #[test]
    fn wide_neq_of_identical_const_arrays_is_bounded_safe() {
        let net = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 a
5 state 3 b
6 zero 2
7 init 3 4 6
8 init 3 5 6
9 next 3 4 4
10 next 3 5 5
11 sort bitvec 1
12 neq 11 4 5
13 bad 12
";
        match run_depth(net, 2) {
            ArrayBmcOutcome::BoundedNoCex { depth_reached } => assert_eq!(depth_reached, 2),
            other => panic!("expected BoundedNoCex (E2), got {other:?}"),
        }
    }

    /// The PINNED phase-1 shape end-to-end in the lane, tiny domain (E1):
    /// a init 0 written to all-ones in one step vs b init 1 — extensionally
    /// equal at frame 1, so bad = eq(a,b) fires there; this replays ONLY with
    /// the fixed residual-domain word_eq (differing defaults, full cover).
    #[test]
    fn e1_differing_default_full_cover_eq_fires() {
        let net = "\
1 sort bitvec 1
2 sort bitvec 8
3 sort array 1 2
4 state 3 a
5 state 3 b
6 zero 2
7 init 3 4 6
8 one 2
9 init 3 5 8
10 zero 1
11 one 1
12 write 3 4 10 8
13 write 3 12 11 8
14 next 3 4 13
15 next 3 5 5
16 sort bitvec 1
17 eq 16 4 5
18 bad 17
";
        match run_depth(net, 2) {
            ArrayBmcOutcome::Unsafe { depth, fired, .. } => {
                assert_eq!(depth, 1);
                assert_eq!(fired, vec![0]);
            }
            other => panic!("expected Unsafe@1, got {other:?}"),
        }
    }

    // -- k-induction (phase-2 stepping stone) -----------------------------------

    fn run_kind(net: &str, max_k: usize) -> ArrayKindOutcome {
        let prog = parse(net).expect("parse");
        check_array_kinduction(
            &prog,
            &ArrayKindConfig {
                max_k,
                ..ArrayKindConfig::default()
            },
        )
    }

    /// SAFE wide net (write 5 forever, bad looks for 6): 1-inductive — the
    /// step query closes via the ROW axiom, and the independent LRAT-checked
    /// discharge verifies both base and step, minting the ONLY unbounded-safe
    /// verdict this engine can produce.
    #[test]
    fn kinduction_proves_wide_safe_net() {
        let net = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
8 zero 1
10 constd 2 5
13 write 3 4 8 10
7 next 3 4 13
9 read 2 4 8
12 sort bitvec 1
15 constd 2 6
11 eq 12 9 15
14 bad 11
";
        match run_kind(net, 4) {
            ArrayKindOutcome::ProvedSafe { k } => assert!(k <= 2, "expected small k, got {k}"),
            other => panic!("expected ProvedSafe, got {other:?}"),
        }
    }

    /// UNSAFE wide net: the base BMC inside the k-induction loop must find
    /// the depth-1 counterexample (replay-gated) before any step reasoning
    /// can mislead.
    #[test]
    fn kinduction_unsafe_finds_base_cex() {
        let net = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
8 zero 1
10 constd 2 5
13 write 3 4 8 10
7 next 3 4 13
9 read 2 4 8
12 sort bitvec 1
11 eq 12 9 10
14 bad 11
";
        match run_kind(net, 4) {
            ArrayKindOutcome::Unsafe { depth, fired, .. } => {
                assert_eq!(depth, 1);
                assert_eq!(fired, vec![0]);
            }
            other => panic!("expected Unsafe@1, got {other:?}"),
        }
    }

    /// GENUINELY 2-inductive (verified by hand): the saturating 2-bit
    /// counter `c' = ite(c==2, c, c+1)` can never be 3 AFTER a step, and
    /// not-bad@1 pins mem[3]=0 (1-bit elements), so bad@2 is contradictory —
    /// while at k=1 a free c0=3 fires bad@1, so k=1 must NOT close. The lane
    /// must find exactly k=2.
    #[test]
    fn kinduction_proves_saturating_counter_at_k2() {
        let net = "\
1 sort bitvec 16
2 sort bitvec 1
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
22 sort bitvec 2
7 state 22 c
8 zero 22
9 init 22 7 8
10 one 22
11 add 22 7 10
24 constd 22 2
25 eq 2 7 24
26 ite 22 25 7 11
12 next 22 7 26
13 one 2
23 uext 1 7 14
14 write 3 4 23 13
15 next 3 4 14
16 constd 1 3
17 read 2 4 16
18 bad 17
";
        match run_kind(net, 4) {
            ArrayKindOutcome::ProvedSafe { k } => assert_eq!(k, 2),
            other => panic!("expected ProvedSafe@2, got {other:?}"),
        }
    }

    /// SAFE but NOT k-inductive within max_k (no invariant strengthening
    /// yet): a WIDE (16-bit) saturating counter walks 0,1,2 and stops; bad
    /// reads mem[100]. Concretely safe (index 100 is never written), but a
    /// FREE frame-0 counter can start at 100-k+1 and reach 100 in the step
    /// window at EVERY k. The honest outcome is BoundedNoCex, NEVER
    /// ProvedSafe.
    #[test]
    fn kinduction_honest_bounded_when_not_inductive() {
        let net = "\
1 sort bitvec 16
2 sort bitvec 1
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
7 state 1 i
8 zero 1
9 init 1 7 8
10 one 1
11 add 1 7 10
12 constd 1 2
13 eq 2 7 12
14 ite 1 13 7 11
15 next 1 7 14
16 one 2
17 write 3 4 7 16
18 next 3 4 17
19 constd 1 100
20 read 2 4 19
21 bad 20
";
        match run_kind(net, 3) {
            ArrayKindOutcome::BoundedNoCex { depth_reached } => {
                assert_eq!(depth_reached, 2, "base must have closed through k-1");
            }
            other => panic!("expected BoundedNoCex (honest non-verdict), got {other:?}"),
        }
    }

    /// Extensionality inside the step query: two held arrays with bad =
    /// neq(a, b) — NOT-bad at frame 0 assumes a==b, both hold, so bad at
    /// frame 1 contradicts the same (deduplicated) eq entry. 1-inductive,
    /// independently discharged.
    #[test]
    fn kinduction_eq_hold_net_proves() {
        let net = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 a
5 state 3 b
6 zero 2
7 init 3 4 6
8 init 3 5 6
9 next 3 4 4
10 next 3 5 5
11 sort bitvec 1
12 neq 11 4 5
13 bad 12
";
        match run_kind(net, 4) {
            ArrayKindOutcome::ProvedSafe { k } => assert_eq!(k, 1),
            other => panic!("expected ProvedSafe@1, got {other:?}"),
        }
    }

    // -- decline rules ---------------------------------------------------------

    /// Mixed-dims and array-vs-scalar equality stay declined (fail-closed).
    #[test]
    fn declines_mixed_dims_array_equality() {
        let net = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 sort bitvec 4
5 sort array 1 4
6 state 3 a
7 state 5 b
8 zero 2
9 init 3 6 8
10 zero 4
11 init 5 7 10
12 next 3 6 6
13 next 5 7 7
14 sort bitvec 1
15 eq 14 6 7
16 bad 15
";
        match run(net) {
            ArrayBmcOutcome::Declined { reason } => {
                assert!(reason.contains("mixed-dims"), "reason: {reason}");
            }
            other => panic!("expected Declined, got {other:?}"),
        }
    }

    #[test]
    fn declines_no_next_state() {
        let net = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 mem
5 zero 1
6 read 2 4 5
7 sort bitvec 1
8 redor 7 6
9 bad 8
";
        match run(net) {
            ArrayBmcOutcome::Declined { reason } => {
                assert!(reason.contains("no `next`"), "reason: {reason}");
            }
            other => panic!("expected Declined, got {other:?}"),
        }
    }

    #[test]
    fn declines_scalar_only_net() {
        let net = "\
1 sort bitvec 8
2 state 1 c
3 zero 1
4 init 1 2 3
5 one 1
6 add 1 2 5
7 next 1 2 6
8 sort bitvec 1
9 constd 1 3
10 eq 8 2 9
11 bad 10
";
        match run(net) {
            ArrayBmcOutcome::Declined { reason } => {
                assert!(reason.contains("no array state"), "reason: {reason}");
            }
            other => panic!("expected Declined, got {other:?}"),
        }
    }

    #[test]
    fn declines_array_input() {
        let net = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 input 3 mem
5 state 3 m2
6 next 3 5 5
7 zero 2
8 init 3 5 7
9 zero 1
10 read 2 4 9
11 sort bitvec 1
12 redor 11 10
13 bad 12
";
        match run(net) {
            ArrayBmcOutcome::Declined { reason } => {
                assert!(reason.contains("array-sorted input"), "reason: {reason}");
            }
            other => panic!("expected Declined, got {other:?}"),
        }
    }
}
