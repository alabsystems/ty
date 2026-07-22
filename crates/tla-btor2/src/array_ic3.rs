// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! IC3/PDR frames over the lazy-array SSA/epoch encoding (phase 3) — the
//! frame-strengthening engine the k-induction lane provably needs on nets
//! whose property is not k-inductive with a free frame-0 array (measured:
//! array_lt200 and simple-stack-pred1 are step-SAT at every k <= 12).
//!
//! # Vocabulary
//!
//! Lemmas are clauses over FRAME-BOUNDARY BITS:
//!
//! * bit-literals of scalar state variables, and
//! * bit-literals of READ PINS `p_j = read(A, c_j)` at CONSTANT indices `c_j`
//!   discovered adaptively — whenever a chain-consistent predecessor
//!   extraction pins nondet root cells at concrete indices, those indices
//!   join the probe set (CEGAR over probes; the probe budget is DERIVED from
//!   the read budget, never a separate magic constant).
//!
//! # Queries (all through the existing [`crate::array_bmc`] machinery)
//!
//! One persistent 1-step [`Unroller`] in `free_init` mode (the k-induction
//! step object at k = 1) serves consecution/blocking/propagation; initiation
//! is checked SYNTACTICALLY and EXACTLY (this lane's preflight restricts
//! `init` to constants, so a cube intersects Init iff no literal contradicts
//! a constant init bit — no solver involved). Per query: the cached
//! bit-blasted base circuit CNF, plus `F_i` lemmas emitted as plain CNF
//! clauses over frame-0 input bits, plus unit-clause cube assumptions
//! (neg-cube at 0, primed cube at 1, Bad mirror at 0) — then a FRESH ay-sat
//! solve (never incremental: the documented `tla-aiger/src/ic3/config.rs`
//! hazard is never re-entered). Every SAT model runs
//! [`refine_or_extract`] (with the phase-3 batch frontier unfold) to
//! chain-consistency before it is believed as an abstract predecessor.
//!
//! # The invariant object and the validation triple
//!
//! On convergence (`delta_i` empty), the frames serialize into a standalone
//! [`ArrayFrameInvariant`] over ORIGINAL-net state variables and pinned reads
//! — no Unroller state, no engine solver state. `ProvedSafe` is minted ONLY
//! after the `validate.rs`-style triple
//!
//! ```text
//!   I: Init            ∧ ¬Inv    UNSAT
//!   C: Inv ∧ T ∧ constr ∧ ¬Inv'  UNSAT
//!   S: Inv ∧ constr    ∧ Bad     UNSAT
//! ```
//!
//! is discharged INDEPENDENTLY of the frames engine, every leaf LRAT-checked
//! via [`crate::array_cert::solve_dimacs_unsat_lrat`] (ay-sat proof mode +
//! the separate `ay-lrat-check` crate):
//!
//! * **Tier A** (flattenable arrays): the invariant is translated to a
//!   [`ChcExpr`] over the ORIGINAL program's [`VcComponents`] and discharged
//!   through [`crate::array_cert`]'s ground bit-level route — zero
//!   epoch/Unroller code in the trust path. Mandatory whenever the net's
//!   array states are structurally flattenable; its result is FINAL there.
//! * **Tier B** (wide arrays, flattening impossible in principle): a FRESH
//!   one-step encoding built by a NEW [`Unroller`] (no inherited refinement
//!   history) with a CLOSED-FORM EAGER instantiation policy — every read's
//!   full ROW/select-over-ite spine to the root plus ALL root congruence
//!   pairs. Soundness is one-directional and exactly the k-induction-step
//!   argument: each check is a NO-MODEL claim, the lazy-read encoding with
//!   ANY axiom subset OVER-approximates the true one-step semantics, so an
//!   LRAT-verified UNSAT of the over-approximation soundly discharges the
//!   true VC on the original net semantics.
//!
//! Any SAT/inconclusive/budget outcome DOWNGRADES the claim to
//! [`ArrayIc3Outcome::BoundedNoCex`] — never reported unbounded.
//!
//! # Counterexample side (fail-closed)
//!
//! An obligation chain reaching frame 0 is only a CANDIDATE (cubes are
//! projections; concatenation is not a real trace). It is confirmed by
//! running the existing replay-gated BMC loop to the candidate depth: the
//! ONLY `Unsafe` this lane can mint carries a `word_replay`-validated model —
//! the identical gate as the BMC lane. An unconfirmed candidate is an honest
//! [`ArrayIc3Outcome::Declined`].

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use ay_chc::{ChcExpr, ChcOp, ChcSort, ChcVar};
use std::sync::Arc;

use crate::array_bmc::{
    add_root_congruence_pairs, array_dims, check_array_bmc, circuit_to_cnf, discharge_unsat_lrat,
    instantiate_structural, lane_supported, lit_to_dimacs, mask, refine_or_extract, ATerm,
    ArrayBmcConfig, ArrayBmcOutcome, CandidatePlan, InputRole, Unroller,
};
use crate::bitblast::{bitblast, BitblastedCircuit};
use crate::to_chc::translate_to_chc_with_vc;
use crate::types::{Btor2Node, Btor2Program, Btor2Sort, NodeId};
use crate::witness::Btor2Witness;
use crate::word_replay::{eval_init_state_values, WordLevelModel, WordValue};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Configuration for the array IC3/PDR lane.
#[derive(Debug, Clone)]
pub struct ArrayIc3Config {
    /// Maximum number of frames before giving up (bounded fact only).
    pub max_frames: usize,
    /// Maximum refinement iterations per SAT query.
    pub max_refinements_per_query: usize,
    /// Read-table budget (the probe budget is derived from this).
    pub max_reads: usize,
    /// Wall-clock budget for the whole lane; `None` = unbounded.
    pub time_budget: Option<Duration>,
    /// Print progress to stderr.
    pub verbose: bool,
}

impl Default for ArrayIc3Config {
    fn default() -> Self {
        ArrayIc3Config {
            max_frames: 40,
            max_refinements_per_query: 64,
            max_reads: 512,
            time_budget: None,
            verbose: false,
        }
    }
}

/// A literal of the serialized frame invariant: `atom == positive`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvLit {
    /// The frame-boundary bit.
    pub atom: InvAtom,
    /// `true` = the bit is 1.
    pub positive: bool,
}

/// A frame-boundary bit of the ORIGINAL net: a scalar state bit or a bit of
/// a pinned constant-index array read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvAtom {
    /// Bit `bit` of scalar state `state`.
    StateBit {
        /// BTOR2 state node id.
        state: NodeId,
        /// Bit position (LSB = 0).
        bit: u32,
    },
    /// Bit `bit` of `read(probes[probe].0, probes[probe].1)`.
    ProbeBit {
        /// Index into [`ArrayFrameInvariant::probes`].
        probe: usize,
        /// Bit position (LSB = 0).
        bit: u32,
    },
    /// Bit `bit` of the UNIVERSAL cell `A[ι]` of the Λ-pinned array state
    /// `lambdas[lambda]`: a clause containing `UCellBit` atoms is a
    /// ∀-cell fact (`forall i: clause[A[i]/A[ι]]`). Engine-side the pin is a
    /// read at a fresh shared free-input index; validation-side Tier A
    /// expands the ∀ into all `2^iw` ground instances and Tier B
    /// instantiates premise occurrences over a closed-form index set while
    /// giving each ¬Inv occurrence a fresh free witness index.
    UCellBit {
        /// Index into [`ArrayFrameInvariant::lambdas`].
        lambda: usize,
        /// Bit position (LSB = 0).
        bit: u32,
    },
}

/// The standalone inductive-invariant artifact the frames converge to. It
/// references ONLY original-net state variables and pinned constant-index
/// reads — no Unroller state, no engine solver state — so the validation
/// triple can be discharged against a completely fresh encoding.
#[derive(Debug, Clone)]
pub struct ArrayFrameInvariant {
    /// The probe set: `(array state id, constant index)` per pin.
    pub probes: Vec<(NodeId, u128)>,
    /// The Λ-pin set: `(array state id, index width)` per universal-cell
    /// pin. A clause's [`InvAtom::UCellBit`] literals quantify over the
    /// cells of `lambdas[lambda].0`.
    pub lambdas: Vec<(NodeId, u32)>,
    /// CNF clauses over the vocabulary (conjunction of disjunctions).
    pub clauses: Vec<Vec<InvLit>>,
}

/// Which validation tier discharged the triple.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayCertTier {
    /// Tier A: original net fully flattened through the `array_cert` ground
    /// bit-level route (`to_chc` VC components; no epoch/Unroller code).
    FlattenedLrat,
    /// Tier B: fresh one-step epoch encoding with the closed-form eager
    /// instantiation policy (over-approximation UNSAT direction).
    EagerOneStepLrat,
}

/// Outcome of the array IC3 lane.
#[derive(Debug)]
pub enum ArrayIc3Outcome {
    /// UNBOUNDED SAFE: the frames converged AND the serialized invariant's
    /// triple (I/C/S) was discharged independently, every leaf LRAT-checked.
    /// This is the only variant that is an unsat verdict.
    ProvedSafe {
        /// Frame level at which `F_i == F_{i+1}`.
        converged_at: usize,
        /// The validated invariant artifact.
        invariant: ArrayFrameInvariant,
        /// Which tier discharged the triple.
        tier: ArrayCertTier,
    },
    /// A concrete counterexample, CONFIRMED by the replay-gated BMC loop
    /// (identical gate as [`ArrayBmcOutcome::Unsafe`]).
    Unsafe {
        /// Frame index at which the bad property fires.
        depth: usize,
        /// Indices into `program.bad_properties` that fire (from replay).
        fired: Vec<usize>,
        /// The replay-validated concrete model.
        model: WordLevelModel,
        /// btorsim witness when serializable (fail-closed `None`).
        witness: Option<Btor2Witness>,
    },
    /// No verdict: frames did not converge within budget (or the triple did
    /// not discharge — the claim DOWNGRADES here). A bounded fact only.
    BoundedNoCex {
        /// Number of fully blocked frame levels.
        frames_completed: usize,
    },
    /// Outside the lane's slice or a budget/cap decline. The caller's
    /// default decision tree proceeds unchanged.
    Declined {
        /// Why the lane declined.
        reason: String,
    },
}

/// Run the array IC3/PDR lane. See the module docs for the frame vocabulary,
/// the query discipline, and the fail-closed validation gating.
#[must_use]
pub fn check_array_ic3(program: &Btor2Program, config: &ArrayIc3Config) -> ArrayIc3Outcome {
    match ic3_inner(program, config) {
        Ok(outcome) => outcome,
        Err(reason) => ArrayIc3Outcome::Declined { reason },
    }
}

// ---------------------------------------------------------------------------
// Lemma / frame data structures (the tla-aiger ic3/frame.rs template:
// sorted-literal cubes, forward + backward subsumption, delta encoding)
// ---------------------------------------------------------------------------

/// Packed literal over the engine vocabulary: `atom << 1 | (bit == 0)`.
type PLit = u32;

fn plit(atom: usize, bit_is_one: bool) -> PLit {
    ((atom as u32) << 1) | u32::from(!bit_is_one)
}
fn plit_atom(l: PLit) -> usize {
    (l >> 1) as usize
}
fn plit_is_one(l: PLit) -> bool {
    l & 1 == 0
}

/// `a ⊆ b` for sorted literal vectors.
fn sorted_subset(a: &[PLit], b: &[PLit]) -> bool {
    let mut i = 0;
    for &lb in b {
        if i == a.len() {
            return true;
        }
        if a[i] == lb {
            i += 1;
        }
    }
    i == a.len()
}

/// A lemma stored as the (sorted) CUBE it blocks; as a clause it is the
/// negation. Lemma L1 subsumes L2 (as clauses) iff cube(L1) ⊆ cube(L2).
#[derive(Clone)]
struct Lemma {
    cube: Vec<PLit>,
}

/// Delta-encoded frames: `lemmas[l]` holds lemmas whose highest proven level
/// is `l`; the clause set of `F_i` is the union of `lemmas[l]` for `l >= i`.
struct Frames {
    lemmas: Vec<Vec<Lemma>>,
}

impl Frames {
    fn new() -> Self {
        // Level 0 (Init) carries no lemmas; start with level 1.
        Frames {
            lemmas: vec![Vec::new(), Vec::new()],
        }
    }

    fn top(&self) -> usize {
        self.lemmas.len() - 1
    }

    fn push_new(&mut self) {
        self.lemmas.push(Vec::new());
    }

    /// All clauses of `F_i` (levels >= i).
    fn clauses_at(&self, i: usize) -> impl Iterator<Item = &Lemma> {
        self.lemmas[i.min(self.lemmas.len() - 1)..].iter().flatten()
    }

    /// Is `cube` already blocked at level `i` (some lemma at level >= i
    /// subsumes it)?
    fn is_blocked(&self, i: usize, cube: &[PLit]) -> bool {
        self.clauses_at(i).any(|l| sorted_subset(&l.cube, cube))
    }

    /// Add a lemma blocking `cube` at `level`, with forward subsumption
    /// (skip if an existing lemma at >= level already subsumes it) and
    /// backward subsumption (drop existing lemmas at <= level the new one
    /// subsumes). Returns whether it was added.
    fn add_lemma(&mut self, level: usize, cube: Vec<PLit>) -> bool {
        if self.is_blocked(level, &cube) {
            return false;
        }
        for l in 1..=level.min(self.lemmas.len() - 1) {
            self.lemmas[l].retain(|old| !sorted_subset(&cube, &old.cube));
        }
        self.lemmas[level].push(Lemma { cube });
        true
    }
}

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

enum VarKind {
    Scalar,
    Probe {
        /// Constant index of the pin.
        index: u128,
        /// Position in the serialized probe list.
        probe_idx: usize,
    },
    /// Universal-cell pin `A[ι]` at the array's shared fresh free index ι:
    /// cubes over Λ-bits block `∃ cell matching pattern`, so their clauses
    /// are ∀-cell facts. Λ vars never enter model cubes (a model only
    /// witnesses ONE cell); they enter lemmas via explicit Λ-projection.
    Lambda {
        /// Position in the serialized lambda list.
        lambda_idx: usize,
    },
}

/// One vocabulary variable: a scalar state or a probe pin, with its input
/// positions (into each unroller's input-role order) at the frame boundaries.
struct VarEntry {
    sid: NodeId,
    kind: VarKind,
    width: u32,
    /// Input index (== role index) of the frame-0 value in the STEP unroller.
    in0: usize,
    /// Input index of the frame-1 value in the STEP unroller.
    in1: usize,
    /// Input index of the frame-0 value in the BAD unroller (`Engine::ub`).
    in0_bad: usize,
    /// First atom id; atoms `atom_base .. atom_base + width`.
    atom_base: usize,
}

/// Which persistent unroller a query runs on: the 1-step consecution object
/// or the depth-0 bad-state object (see [`Engine::ub`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum UWhich {
    Step,
    Bad,
}

/// Which frame boundary a vocabulary literal is rendered at.
#[derive(Clone, Copy, PartialEq, Eq)]
enum VFrame {
    /// Step unroller, frame 0.
    S0,
    /// Step unroller, frame 1.
    S1,
    /// Bad unroller, frame 0.
    B0,
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Cached bit-blast of the engine unroller (invalidated whenever refinement
/// or probe addition grows the unrolled program).
struct Built {
    version: (usize, usize),
    /// Structurally UNSAT (constant-false constraint): every query is UNSAT.
    structurally_unsat: bool,
    num_vars: usize,
    /// The base conjunction (AND-gate Tseitin + constraint clauses), shared
    /// by every query at this version.
    base_clauses: Vec<Vec<i32>>,
    /// AIG literals of every input's bits, in input-role order.
    input_bits: Vec<Vec<u64>>,
}

/// A single engine query over the persistent 1-step unroller.
#[derive(Default)]
struct Query<'q> {
    /// Assert every clause of `F_i` at frame 0 (lemma cubes, negated).
    frame_level: Option<usize>,
    /// Assert Init at frame 0 (unit clauses on constant init bits — the
    /// PARTIAL, weaker-than-real Init; sound for UNSAT-side use because a
    /// weaker premise only over-approximates predecessors).
    assert_init: bool,
    /// Assert `¬cube` at frame 0 (one clause).
    neg_cube0: Option<&'q [PLit]>,
    /// Assert `cube` at frame 1 (unit clauses over frame-1 bits).
    cube1: Option<&'q [PLit]>,
    /// Assert `OR(bad)` at frame 0 (unit clause on the bad mirror bit).
    bad0: bool,
}

enum QueryOutcome {
    Unsat,
    /// Chain-consistent model: decoded input values + candidate plan.
    Model(Vec<u128>, CandidatePlan),
}

enum BlockResult {
    Blocked,
    /// An obligation chain reached Init: candidate counterexample of the
    /// given depth (bad fires at that frame index).
    CexCandidate(usize),
}

struct Engine<'a> {
    program: &'a Btor2Program,
    /// The persistent 1-step unroller (`T ∧ constr@0..1` baked): consecution,
    /// blocking and propagation queries — the exact triple-C shape.
    u: Unroller<'a>,
    /// The persistent DEPTH-0 bad unroller (`constr@0` only): the bad query
    /// `F_top ∧ constr@0 ∧ Bad@0` — the exact triple-S shape. BTOR2 traces
    /// may END at the bad frame, so requiring a legal successor (the phase-3
    /// bad query through `u`) under-approximated Bad: on constraint-dead-end
    /// nets it was UNSAT at every level, frames converged trivially, and the
    /// triple gate had to reject the junk invariant (pure completeness loss).
    ub: Unroller<'a>,
    vars: Vec<VarEntry>,
    /// atom id -> (var index, bit).
    atoms: Vec<(usize, u32)>,
    probes: Vec<(NodeId, u128)>,
    probe_set: HashSet<(NodeId, u128)>,
    max_probes: usize,
    /// Λ pins: `(array state id, index width)` per universal-cell pin,
    /// aligned with `VarKind::Lambda::lambda_idx`.
    lambdas: Vec<(NodeId, u32)>,
    /// Input index (in `ub`) of the 1-bit mirror of `OR(bad)@0`.
    bad0_in: usize,
    frames: Frames,
    built: Option<Built>,
    built_bad: Option<Built>,
    /// Exact constant init values (states with `init` only).
    init_vals: HashMap<NodeId, WordValue>,
    max_refinements: usize,
    max_reads: usize,
    /// Version watermarks (reads-table lengths) at the last eager structural
    /// closure per unroller — `eager_close` is a no-op while unchanged.
    closed_reads_step: usize,
    closed_reads_bad: usize,
    /// STEP-2 (lt200 loop A): per pinned array, the PREMISE-instance read roles
    /// at which every Λ-cell frame lemma is ALSO rendered — reads of `A@frame`
    /// at the chain's write/read address terms (the symbolic indices the query
    /// cone actually mentions). `(frame0_roles, frame1_roles)` for the step
    /// unroller; frame-0 roles for the bad unroller. Rebuilt by `eager_close`.
    lam_prem_step: HashMap<NodeId, (Vec<usize>, Vec<usize>)>,
    lam_prem_bad: HashMap<NodeId, Vec<usize>>,
    verbose: bool,
    start: Instant,
    budget: Option<Duration>,
    /// Monotone obligation sequence number (heap tie-break).
    seq: u64,
}

impl<'a> Engine<'a> {
    fn new(
        program: &'a Btor2Program,
        config: &ArrayIc3Config,
        init_vals: HashMap<NodeId, WordValue>,
        start: Instant,
    ) -> Result<Self, String> {
        let mut u = Unroller::new(program, config.max_reads, true)?;
        u.seed_through(1)?;
        u.emit_constraints_through(1)?;

        // The depth-0 bad unroller: free init, frame 0 only, constr@0 only.
        // Its query shape (`F ∧ constr@0 ∧ Bad@0`) mirrors triple-S exactly.
        let mut ub = Unroller::new(program, config.max_reads, true)?;
        ub.emit_constraints_through(0)?;

        // OR of all bad conditions at frame 0 of the BAD unroller, mirrored
        // so its concrete value is an input bit assertable in the bad query.
        let mut bad0: Option<i64> = None;
        for i in 0..program.bad_properties.len() {
            let bid = program.bad_properties[i];
            let cond = ub.node_cond(bid)?;
            let cu = ub.unroll_ref(0, cond)?;
            bad0 = Some(match bad0 {
                None => cu,
                Some(acc) => ub.emit(Btor2Node::Or, 1, vec![acc, cu]),
            });
        }
        let bad0 = bad0.ok_or("no bad properties")?;
        let (_, bad0_in) = ub.add_mirror(bad0, 1);

        // Scalar-state vocabulary: frame-0 value is the free-init fresh
        // input; frame-1 value is mirrored; the bad unroller's frame-0 value
        // is its own free-init input.
        let mut vars = Vec::new();
        let mut atoms = Vec::new();
        let state_ids = u.state_ids.clone();
        for sid in state_ids {
            let Some(u1) = u.scalar_state_at(1, sid) else {
                continue; // array state — probe pins cover it adaptively
            };
            let line = program
                .lines
                .iter()
                .find(|l| l.id == sid)
                .ok_or("state line missing")?;
            let width = match program.sorts.get(&line.sort_id) {
                Some(Btor2Sort::BitVec(w)) => *w,
                _ => return Err("scalar state with non-bitvector sort".into()),
            };
            let find_st0 = |roles: &[InputRole]| {
                roles
                    .iter()
                    .position(|r| matches!(r, InputRole::Frame0State { src } if *src == sid))
                    .ok_or("frame-0 state input missing (free_init invariant)")
            };
            let in0 = find_st0(&u.input_roles)?;
            let in0_bad = find_st0(&ub.input_roles)?;
            let (_, in1) = u.add_mirror(u1, width);
            let atom_base = atoms.len();
            for b in 0..width {
                atoms.push((vars.len(), b));
            }
            vars.push(VarEntry {
                sid,
                kind: VarKind::Scalar,
                width,
                in0,
                in1,
                in0_bad,
                atom_base,
            });
        }

        // Λ-pin universal-cell vocabulary — GATED: only array states read at
        // a SYMBOLIC (input-derived, non-constant) index inside the bad
        // property's frame-0 combinational cone get a ∀-cell pin. An array
        // the property does not quantify over (no symbolic read in the bad
        // cone — e.g. simple-stack, whose bad cone is two scalar flags) gets
        // ZERO Λ pins, so its Λ path stays inert and the engine runs the
        // exact const-pin lane on it. The gate is derived from net structure
        // (no magic constant); soundness never rests on it — every Λ lemma is
        // re-checked by the independent triple gate. Each eligible array gets
        // one pin pair at a FRESH shared free-input index ι: r0 = read(A@0,ι)
        // and r1 = read(A@1,ι) on the SAME ι (step unroller), plus a frame-0
        // pin at the bad unroller's own free index. A cube over Λ-bits blocks
        // `∃ cell matching pattern`; its clause is a ∀-cell fact (consecution
        // soundness: ι is a free input in a no-model claim).
        let lam_eligible = lambda_eligible_arrays(program);
        let mut lambdas: Vec<(NodeId, u32)> = Vec::new();
        let array_state_ids = u.state_ids.clone();
        for sid in array_state_ids {
            if !lam_eligible.contains(&sid) {
                continue;
            }
            let Some(t0) = u.array_state_term(0, sid) else {
                continue; // scalar state
            };
            let (iw, ew) = u.terms[t0].dims();
            let t1 = u
                .array_state_term(1, sid)
                .ok_or("array state term missing at frame 1")?;
            let (iota, _) = u.fresh_free_index(iw, "lam");
            let r0 = u.get_or_make_read(t0, iota)?;
            let r1 = u.get_or_make_read(t1, iota)?;
            let t0b = ub
                .array_state_term(0, sid)
                .ok_or("array state term missing in bad unroller")?;
            let (iota_b, _) = ub.fresh_free_index(iw, "lam");
            let r0b = ub.get_or_make_read(t0b, iota_b)?;
            let atom_base = atoms.len();
            for b in 0..ew {
                atoms.push((vars.len(), b));
            }
            vars.push(VarEntry {
                sid,
                kind: VarKind::Lambda {
                    lambda_idx: lambdas.len(),
                },
                width: ew,
                in0: u.reads[r0].var_role,
                in1: u.reads[r1].var_role,
                in0_bad: ub.reads[r0b].var_role,
                atom_base,
            });
            lambdas.push((sid, iw));
        }

        // Probe budget derived from the read budget: each probe consumes two
        // engine reads (both frame boundaries) plus their eager-validation
        // copies and root congruence pairs — a factor-8 reservation keeps the
        // whole probe apparatus within the read budget's order of magnitude.
        let max_probes = (config.max_reads / 8).max(1);

        let mut eng = Engine {
            program,
            u,
            ub,
            vars,
            atoms,
            probes: Vec::new(),
            probe_set: HashSet::new(),
            max_probes,
            lambdas,
            bad0_in,
            frames: Frames::new(),
            built: None,
            built_bad: None,
            init_vals,
            max_refinements: config.max_refinements_per_query,
            max_reads: config.max_reads,
            closed_reads_step: 0,
            closed_reads_bad: 0,
            lam_prem_step: HashMap::new(),
            lam_prem_bad: HashMap::new(),
            verbose: config.verbose,
            start,
            budget: config.time_budget,
            seq: 0,
        };
        // STEP-1 (lt200 divergence fix, loop B): eagerly close the structural
        // array-axiom set ONCE at construction — ROW chains, root congruence
        // pairs, E2 skolems — instead of discovering them one model at a time
        // in every query's refinement loop. Idempotent + version-gated.
        eng.eager_close(UWhich::Step)?;
        eng.eager_close(UWhich::Bad)?;
        Ok(eng)
    }

    /// Eagerly close the unroller's STRUCTURAL array-axiom set (ROW chains via
    /// `instantiate_structural`, all root congruence pairs, E2 skolems) —
    /// delegates to [`eager_instantiate`], which is idempotent (`axiom_done`
    /// flags + congruence-pair dedup) and fail-closed on the read budget (which
    /// transitively bounds congruence pairs at ≤ max_reads²/2 — no new
    /// constant). Version-gated on the reads-table length, so re-closing after
    /// no growth is O(1). The CNF cache needs no manual invalidation:
    /// `ensure_built` is version-keyed and rebuilds automatically.
    ///
    /// This kills lt200-class divergence loop B: with the closure in place the
    /// per-query `refine_or_extract` returns clean on its first iteration for
    /// the equality-free fragment, so queries collapse to probe CEGAR only.
    fn eager_close(&mut self, w: UWhich) -> Result<(), String> {
        let cur = self.unroller(w).reads.len();
        let watermark = match w {
            UWhich::Step => self.closed_reads_step,
            UWhich::Bad => self.closed_reads_bad,
        };
        if watermark == cur && cur != 0 {
            return Ok(());
        }
        let budget = self.max_reads;
        // Fixpoint: structural closure ↔ premise reads (premise reads of A@1
        // spawn ROW chain reads at the SAME addresses, which the next closure
        // pass absorbs; bounded by the read budget, typically 2 passes).
        loop {
            eager_instantiate(self.unroller_mut(w), budget)?;
            let before = self.unroller(w).reads.len();
            self.build_lambda_premise_roles(w)?;
            if self.unroller(w).reads.len() == before {
                break;
            }
        }
        let now = self.unroller(w).reads.len();
        match w {
            UWhich::Step => self.closed_reads_step = now,
            UWhich::Bad => self.closed_reads_bad = now,
        }
        Ok(())
    }

    /// STEP-2 (lt200 loop A — the Λ-lemma RENDERING GAP): build, per pinned
    /// array, PREMISE-instance reads of `A@frame` at every chain write/read
    /// ADDRESS term (the symbolic indices the query cone mentions: the bad
    /// cone's free read address, the step cone's write address), recording
    /// their circuit roles so `negcube_clauses` can render each ∀-cell frame
    /// lemma at those cells too — not just at the ι pin and const probes.
    /// This is the `make_lambda_premises` instantiation, wired into the
    /// PER-QUERY clause rendering (previously certificate-expansion-only).
    /// SOUND: a Λ-cube lemma means `∀ cell: ¬cube[cell]`; rendering at more
    /// cell terms is universal instantiation on the PREMISE side.
    fn build_lambda_premise_roles(&mut self, w: UWhich) -> Result<(), String> {
        let lambdas: Vec<NodeId> = self.lambdas.iter().map(|&(sid, _)| sid).collect();
        for sid in lambdas {
            match w {
                UWhich::Step => {
                    let u = &mut self.u;
                    let chain = lambda_chain(u, sid, &[0, 1]);
                    let idxs = chain_addr_nodes(u, &chain);
                    let mut roles0 = Vec::with_capacity(idxs.len());
                    let mut roles1 = Vec::with_capacity(idxs.len());
                    let t0 = u
                        .array_state_term(0, sid)
                        .ok_or("Λ premise on missing array state @0")?;
                    let t1 = u
                        .array_state_term(1, sid)
                        .ok_or("Λ premise on missing array state @1")?;
                    for &idx in &idxs {
                        let r0 = u.get_or_make_read(t0, idx)?;
                        roles0.push(u.reads[r0].var_role);
                        let r1 = u.get_or_make_read(t1, idx)?;
                        roles1.push(u.reads[r1].var_role);
                    }
                    roles0.sort_unstable();
                    roles0.dedup();
                    roles1.sort_unstable();
                    roles1.dedup();
                    self.lam_prem_step.insert(sid, (roles0, roles1));
                }
                UWhich::Bad => {
                    let ub = &mut self.ub;
                    let chain = lambda_chain(ub, sid, &[0]);
                    let idxs = chain_addr_nodes(ub, &chain);
                    let mut roles0 = Vec::with_capacity(idxs.len());
                    let t0 = ub
                        .array_state_term(0, sid)
                        .ok_or("Λ premise on missing bad array state @0")?;
                    for &idx in &idxs {
                        let r0 = ub.get_or_make_read(t0, idx)?;
                        roles0.push(ub.reads[r0].var_role);
                    }
                    roles0.sort_unstable();
                    roles0.dedup();
                    self.lam_prem_bad.insert(sid, roles0);
                }
            }
        }
        Ok(())
    }

    fn over_budget(&self, what: &str) -> Option<String> {
        self.budget.and_then(|b| {
            (self.start.elapsed() > b).then(|| format!("time budget exceeded ({what})"))
        })
    }

    // -- vocabulary ----------------------------------------------------------

    /// Add probe `(sid, index)`: pin reads at both frame boundaries and one
    /// atom per element bit. Returns false when the budget is exhausted.
    fn add_probe(&mut self, sid: NodeId, index: u128) -> Result<bool, String> {
        if self.probe_set.contains(&(sid, index)) {
            return Ok(false);
        }
        if self.probes.len() >= self.max_probes {
            return Ok(false);
        }
        let t0 = self
            .u
            .array_state_term(0, sid)
            .ok_or("array state term missing at frame 0")?;
        let t1 = self
            .u
            .array_state_term(1, sid)
            .ok_or("array state term missing at frame 1")?;
        let (iw, ew) = self.u.terms[t0].dims();
        let cidx = self.u.const_index(iw, index & mask(iw));
        let r0 = self.u.get_or_make_read(t0, cidx)?;
        let r1 = self.u.get_or_make_read(t1, cidx)?;
        let in0 = self.u.reads[r0].var_role;
        let in1 = self.u.reads[r1].var_role;
        let t0b = self
            .ub
            .array_state_term(0, sid)
            .ok_or("array state term missing in bad unroller")?;
        let cidx_b = self.ub.const_index(iw, index & mask(iw));
        let r0b = self.ub.get_or_make_read(t0b, cidx_b)?;
        let in0_bad = self.ub.reads[r0b].var_role;
        let probe_idx = self.probes.len();
        self.probes.push((sid, index));
        self.probe_set.insert((sid, index));
        let atom_base = self.atoms.len();
        for b in 0..ew {
            self.atoms.push((self.vars.len(), b));
        }
        self.vars.push(VarEntry {
            sid,
            kind: VarKind::Probe { index, probe_idx },
            width: ew,
            in0,
            in1,
            in0_bad,
            atom_base,
        });
        if self.verbose {
            eprintln!("array-ic3: probe added: state {sid}[{index}] ({ew} bits)");
        }
        // Close the new pin reads' structural axioms immediately (both
        // unrollers gained reads) — see `eager_close`.
        self.eager_close(UWhich::Step)?;
        self.eager_close(UWhich::Bad)?;
        Ok(true)
    }

    /// Exact syntactic initiation: a cube intersects Init iff no literal
    /// contradicts a constant init bit (free bits are unconstrained — Init
    /// is a product of constants and free variables in this lane's slice).
    ///
    /// Λ-literals are checked GROUP-WISE per pinned array: the ∃-cell
    /// pattern intersects Init iff SOME initial cell value satisfies every
    /// Λ-bit of that array — the const-init default (when not every index
    /// is explicitly overridden) or any explicit init cell. Nondet-init
    /// arrays always intersect (correctly refusing Λ over-generalization).
    fn cube_intersects_init(&self, cube: &[PLit]) -> bool {
        // (lambda var index) -> (must-one mask, must-zero mask).
        let mut lam_pat: HashMap<usize, (u128, u128)> = HashMap::new();
        for &l in cube {
            let (vi, bit) = self.atoms[plit_atom(l)];
            let v = &self.vars[vi];
            let want_one = plit_is_one(l);
            match self.init_vals.get(&v.sid) {
                None => {} // nondeterministic at init — free
                Some(WordValue::Bv { bits, .. }) => {
                    if (bits >> bit) & 1 == 1 && !want_one || (bits >> bit) & 1 == 0 && want_one {
                        return false;
                    }
                }
                Some(WordValue::Array { default, cells, .. }) => match &v.kind {
                    VarKind::Probe { index, .. } => {
                        let cell = cells.get(index).copied().unwrap_or(*default);
                        if (cell >> bit) & 1 == 1 && !want_one || (cell >> bit) & 1 == 0 && want_one
                        {
                            return false;
                        }
                    }
                    VarKind::Lambda { .. } => {
                        let (ones, zeros) = lam_pat.entry(vi).or_insert((0, 0));
                        if want_one {
                            *ones |= 1u128 << bit;
                        } else {
                            *zeros |= 1u128 << bit;
                        }
                    }
                    VarKind::Scalar => {}
                },
            }
        }
        for (vi, (ones, zeros)) in lam_pat {
            let v = &self.vars[vi];
            let Some(WordValue::Array { default, cells, .. }) = self.init_vals.get(&v.sid) else {
                continue; // nondet init: free — intersects
            };
            let VarKind::Lambda { lambda_idx } = v.kind else {
                continue;
            };
            let iw = self.lambdas[lambda_idx].1;
            let matches = |val: u128| val & ones == ones && val & zeros == 0;
            // The default is a live candidate unless EVERY index is
            // explicitly overridden (conservative: u128 domain compare).
            let default_live = (iw >= 128) || (cells.len() as u128) < (1u128 << iw);
            let mut hit = default_live && matches(*default);
            if !hit {
                hit = cells.values().any(|&c| matches(c));
            }
            if !hit {
                return false; // no initial cell can satisfy the ∃ pattern
            }
        }
        true
    }

    // -- circuit / CNF -------------------------------------------------------

    fn unroller(&self, w: UWhich) -> &Unroller<'a> {
        match w {
            UWhich::Step => &self.u,
            UWhich::Bad => &self.ub,
        }
    }

    fn unroller_mut(&mut self, w: UWhich) -> &mut Unroller<'a> {
        match w {
            UWhich::Step => &mut self.u,
            UWhich::Bad => &mut self.ub,
        }
    }

    fn built_slot(&mut self, w: UWhich) -> &mut Option<Built> {
        match w {
            UWhich::Step => &mut self.built,
            UWhich::Bad => &mut self.built_bad,
        }
    }

    fn built_ref(&self, w: UWhich) -> &Built {
        match w {
            UWhich::Step => self.built.as_ref().expect("built"),
            UWhich::Bad => self.built_bad.as_ref().expect("built"),
        }
    }

    fn version(&self, w: UWhich) -> (usize, usize) {
        let u = self.unroller(w);
        (u.lines_len(), u.constraints_len())
    }

    fn ensure_built(&mut self, w: UWhich) -> Result<(), String> {
        let version = self.version(w);
        if self
            .built_slot(w)
            .as_ref()
            .is_some_and(|b| b.version == version)
        {
            return Ok(());
        }
        // Sentinel constant-true Bad line: circuit_to_cnf then contributes
        // exactly the constraint + AND-gate clauses (bad folds trivially
        // true), which is the base conjunction every IC3 query shares.
        let u = self.unroller_mut(w);
        let one = u.emit(Btor2Node::One, 1, vec![]);
        let program = u.assemble_program(&[one]);
        let circuit = bitblast(&program, 128).map_err(|e| format!("bit-blast: {e}"))?;
        let input_bits: Vec<Vec<u64>> = circuit
            .input_bits
            .iter()
            .map(|(_, bits)| bits.clone())
            .collect();
        let version = self.version(w);
        let built = match circuit_to_cnf(&circuit) {
            None => Built {
                version,
                structurally_unsat: true,
                num_vars: 0,
                base_clauses: Vec::new(),
                input_bits,
            },
            Some((num_vars, clauses)) => Built {
                version,
                structurally_unsat: false,
                num_vars,
                base_clauses: clauses,
                input_bits,
            },
        };
        *self.built_slot(w) = Some(built);
        Ok(())
    }

    /// DIMACS literal of `bit` of input `in_idx` under the current build:
    /// `Ok(dimacs)` or `Err(constant_value)` for a folded-constant bit.
    fn bit_lit(built: &Built, in_idx: usize, bit: u32) -> Result<i32, bool> {
        let lit = built
            .input_bits
            .get(in_idx)
            .and_then(|bits| bits.get(bit as usize))
            .copied()
            .unwrap_or(0);
        match lit {
            0 => Err(false),
            1 => Err(true),
            l => Ok(lit_to_dimacs(l)),
        }
    }

    /// Render one vocabulary literal (`atom == value`) as a DIMACS literal.
    fn vocab_lit(&self, built: &Built, l: PLit, f: VFrame) -> Result<i32, bool> {
        let (vi, bit) = self.atoms[plit_atom(l)];
        let v = &self.vars[vi];
        let in_idx = match f {
            VFrame::S0 => v.in0,
            VFrame::S1 => v.in1,
            VFrame::B0 => v.in0_bad,
        };
        match Self::bit_lit(built, in_idx, bit) {
            Ok(d) => Ok(if plit_is_one(l) { d } else { -d }),
            Err(c) => Err(c == plit_is_one(l)),
        }
    }

    /// Solve one step-unroller query (consecution / propagation shapes).
    fn solve(&mut self, q: &Query<'_>) -> Result<QueryOutcome, String> {
        self.solve_inner(UWhich::Step, q)
    }

    /// The frame-0-only bad query on the DEPTH-0 bad unroller:
    /// `F_level ∧ constr@0 ∧ Bad@0` — exactly the triple-S shape. BTOR2
    /// traces may end at the bad frame, so no successor is required.
    fn solve_bad(&mut self, level: usize) -> Result<QueryOutcome, String> {
        let q = Query {
            frame_level: Some(level),
            bad0: true,
            ..Query::default()
        };
        self.solve_inner(UWhich::Bad, &q)
    }

    /// Solve one query: fresh ay-sat on the cached base CNF plus the query's
    /// clause assumptions, refining spurious models to chain-consistency and
    /// folding newly pinned root cells into the probe set (CEGAR).
    fn solve_inner(&mut self, w: UWhich, q: &Query<'_>) -> Result<QueryOutcome, String> {
        // Frame-1 / init assertions exist only on the step unroller; the
        // bad mirror only on the bad unroller.
        debug_assert!(w == UWhich::Step || (q.cube1.is_none() && !q.assert_init));
        debug_assert!(!q.bad0 || w == UWhich::Bad);
        let f0 = match w {
            UWhich::Step => VFrame::S0,
            UWhich::Bad => VFrame::B0,
        };
        let mut iterations = 0usize;
        loop {
            if let Some(r) = self.over_budget("ic3 query") {
                return Err(r);
            }
            iterations += 1;
            if iterations > self.max_refinements {
                return Err(format!(
                    "refinement iteration cap ({}) hit in ic3 query",
                    self.max_refinements
                ));
            }
            if self.unroller(w).reads.len() > self.max_reads {
                return Err(format!(
                    "read-table cap ({}) exceeded in ic3 query",
                    self.max_reads
                ));
            }
            // Keep the structural closure current across refinement iterations
            // (refinement can add reads); version-gated, O(1) when unchanged.
            self.eager_close(w)?;
            self.ensure_built(w)?;
            let built = self.built_ref(w);
            if built.structurally_unsat {
                return Ok(QueryOutcome::Unsat);
            }

            // ---- assemble the query's extra clauses -------------------------
            let mut extra: Vec<Vec<i32>> = Vec::new();
            let mut trivially_unsat = false;
            let mut push_clause = |cl: Result<Vec<i32>, bool>| match cl {
                Ok(lits) => extra.push(lits),
                Err(true) => {}
                Err(false) => trivially_unsat = true,
            };

            // Lemma clauses of F_i at frame 0: clause = OR of negated cube
            // literals.
            if let Some(level) = q.frame_level {
                let mut clauses: Vec<Result<Vec<i32>, bool>> = Vec::new();
                for lemma in self.frames.clauses_at(level) {
                    self.negcube_clauses(built, &lemma.cube, f0, &mut clauses);
                }
                for cl in clauses {
                    push_clause(cl);
                }
            }
            if let Some(cube) = q.neg_cube0 {
                let mut clauses: Vec<Result<Vec<i32>, bool>> = Vec::new();
                self.negcube_clauses(built, cube, f0, &mut clauses);
                for cl in clauses {
                    push_clause(cl);
                }
            }
            if q.assert_init {
                for cl in self.init_unit_clauses(built) {
                    push_clause(cl);
                }
            }
            if let Some(cube) = q.cube1 {
                for &l in cube {
                    match self.vocab_lit(built, l, VFrame::S1) {
                        Ok(d) => push_clause(Ok(vec![d])),
                        Err(sat) => push_clause(Err(sat)),
                    }
                }
            }
            if q.bad0 {
                match Self::bit_lit(built, self.bad0_in, 0) {
                    Ok(d) => push_clause(Ok(vec![d])),
                    Err(c) => push_clause(Err(c)),
                }
            }
            if trivially_unsat {
                return Ok(QueryOutcome::Unsat);
            }

            // ---- fresh solve -------------------------------------------------
            // Direct fresh `ay_sat::Solver` per query with initial
            // preprocessing DISABLED: IC3 issues many small, structurally
            // near-identical queries, for which the portfolio's sweep
            // preprocessing is pure per-query overhead (measured dominant).
            // The discipline is unchanged — a fresh, non-incremental solver
            // per query, never the documented-unreliable incremental mode;
            // and no engine UNSAT is ever a verdict (the triple validation
            // and cex replay carry their own independent trust paths).
            let built = self.built_ref(w);
            let mut solver = ay_sat::Solver::new(built.num_vars);
            solver.set_preprocess_enabled(false);
            for cl in built.base_clauses.iter().chain(extra.iter()) {
                let lits: Vec<ay_sat::Literal> = cl
                    .iter()
                    .map(|&d| ay_sat::Literal::from_dimacs(d))
                    .collect();
                // A `false` return marks the solver UNSAT internally; solve()
                // below reports it — no separate handling needed.
                let _ = solver.add_clause(lits);
            }
            let model = match solver.solve().into_inner() {
                ay_sat::SatResult::Unsat(_) => return Ok(QueryOutcome::Unsat),
                ay_sat::SatResult::Sat(model) => model,
                _ => return Err("ay-sat returned an inconclusive result".into()),
            };

            // ---- decode + refine to chain-consistency ------------------------
            let built = self.built_ref(w);
            let vals: Vec<u128> = built
                .input_bits
                .iter()
                .map(|bits| {
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
                .collect();
            if vals.len() != self.unroller(w).input_roles.len() {
                return Err(format!(
                    "input decode arity mismatch: {} bit tables vs {} roles",
                    vals.len(),
                    self.unroller(w).input_roles.len()
                ));
            }
            match refine_or_extract(self.unroller_mut(w), &vals) {
                Err(0) => {
                    return Err("ic3 refinement stalled (spurious model, no new axiom)".into())
                }
                Err(_) => continue, // axioms added; version changed; re-solve
                Ok(plan) => {
                    // CEGAR over probes: pin every nondet root cell the
                    // chain-consistent extraction touched at frame 0.
                    let mut added_probe = false;
                    let state_ids = self.unroller(w).state_ids.clone();
                    for sid in state_ids {
                        let Some(t0) = self.unroller(w).array_state_term(0, sid) else {
                            continue;
                        };
                        if !matches!(self.unroller(w).terms[t0], ATerm::RootNondet { .. }) {
                            continue;
                        }
                        let Some(cells) = plan.root_cells.get(&t0) else {
                            continue;
                        };
                        let indices: Vec<u128> = cells.keys().copied().collect();
                        for idx in indices {
                            if self.add_probe(sid, idx)? {
                                added_probe = true;
                            }
                        }
                    }
                    if added_probe {
                        continue; // richer vocabulary; re-solve the query
                    }
                    return Ok(QueryOutcome::Model(vals, plan));
                }
            }
        }
    }

    /// All rendered instances of the clause "¬cube" at frame `f`: the cube's
    /// own atoms (Λ bits at the free-ι pin — the instance that carries the
    /// ∀v generalization argument), PLUS one instance per (Λ group, probe of
    /// the same array) with the Λ bits substituted onto the probe's atoms.
    /// Every extra instance is IMPLIED by the true ∀-clause, so premise-side
    /// use stays sound (a conjunction of implied instances is weaker than
    /// the ∀); they restore EXACTNESS at the probed cells, which is what
    /// keeps the bad/consecution model loop progressing (a returned model's
    /// pinned root cells are all probed by the CEGAR, so it can never
    /// satisfy the rendered clauses while matching a blocked Λ pattern at a
    /// probed cell).
    fn negcube_clauses(
        &self,
        built: &Built,
        cube: &[PLit],
        f: VFrame,
        out: &mut Vec<Result<Vec<i32>, bool>>,
    ) {
        out.push(self.negcube_clause(built, cube, f));
        // Distinct Λ vars present in the cube.
        let mut lam_vis: Vec<usize> = cube
            .iter()
            .filter_map(|&l| {
                let (vi, _) = self.atoms[plit_atom(l)];
                matches!(self.vars[vi].kind, VarKind::Lambda { .. }).then_some(vi)
            })
            .collect();
        lam_vis.sort_unstable();
        lam_vis.dedup();
        for lvi in lam_vis {
            let sid = self.vars[lvi].sid;
            for pv in &self.vars {
                if !matches!(pv.kind, VarKind::Probe { .. }) || pv.sid != sid {
                    continue;
                }
                let subst: Vec<PLit> = cube
                    .iter()
                    .map(|&l| {
                        let (vi, bit) = self.atoms[plit_atom(l)];
                        if vi == lvi {
                            plit(pv.atom_base + bit as usize, plit_is_one(l))
                        } else {
                            l
                        }
                    })
                    .collect();
                out.push(self.negcube_clause(built, &subst, f));
            }
            // STEP-2: render the ∀-cell lemma at the PREMISE-instance cells
            // (reads of the pinned array at the chain's symbolic address
            // terms) — closes the rendering gap that let models place a
            // violating cell at a fresh unprobed address forever (lt200).
            let roles: &[usize] = match f {
                VFrame::S0 => self
                    .lam_prem_step
                    .get(&sid)
                    .map(|(r0, _)| r0.as_slice())
                    .unwrap_or(&[]),
                VFrame::S1 => self
                    .lam_prem_step
                    .get(&sid)
                    .map(|(_, r1)| r1.as_slice())
                    .unwrap_or(&[]),
                VFrame::B0 => self
                    .lam_prem_bad
                    .get(&sid)
                    .map(|r| r.as_slice())
                    .unwrap_or(&[]),
            };
            for &role in roles {
                out.push(self.negcube_clause_lam_at(built, cube, f, lvi, role));
            }
        }
    }

    /// `¬cube` at frame `f`, with the Λ variable `lvi`'s atoms rendered at the
    /// cell read whose circuit role is `role` (a premise instance) instead of
    /// the ι pin. All other literals render through the normal vocabulary.
    fn negcube_clause_lam_at(
        &self,
        built: &Built,
        cube: &[PLit],
        f: VFrame,
        lvi: usize,
        role: usize,
    ) -> Result<Vec<i32>, bool> {
        let mut out = Vec::with_capacity(cube.len());
        for &l in cube {
            let (vi, bit) = self.atoms[plit_atom(l)];
            let r = if vi == lvi {
                match Self::bit_lit(built, role, bit) {
                    Ok(d) => Ok(if plit_is_one(l) { d } else { -d }),
                    Err(c) => Err(c == plit_is_one(l)),
                }
            } else {
                self.vocab_lit(built, l, f)
            };
            match r {
                Ok(d) => out.push(-d),
                Err(true) => {}
                Err(false) => return Err(true),
            }
        }
        if out.is_empty() {
            return Err(false);
        }
        Ok(out)
    }

    /// Clause "¬cube" over frame-`f` bits. `Err(true)` = trivially satisfied.
    fn negcube_clause(&self, built: &Built, cube: &[PLit], f: VFrame) -> Result<Vec<i32>, bool> {
        let mut out = Vec::with_capacity(cube.len());
        for &l in cube {
            match self.vocab_lit(built, l, f) {
                Ok(d) => out.push(-d),
                // Literal constant-true => negation false => drop literal;
                // constant-false => clause satisfied.
                Err(true) => {}
                Err(false) => return Err(true),
            }
        }
        if out.is_empty() {
            // Empty clause: unsatisfiable.
            return Err(false);
        }
        Ok(out)
    }

    /// Unit clauses asserting the (partial) Init at frame 0: constant init
    /// bits of scalar states, probe pins on const-init arrays, and the
    /// AGREEMENT bits of Λ pins on const-init arrays (a bit of `A[ι]` is a
    /// true Init fact for every ι exactly when the default and every explicit
    /// init cell agree on it — conservative when the explicit cells cover the
    /// whole domain, which only weakens the asserted Init).
    fn init_unit_clauses(&self, built: &Built) -> Vec<Result<Vec<i32>, bool>> {
        let mut out = Vec::new();
        for v in &self.vars {
            let (val, care): (u128, u128) = match (self.init_vals.get(&v.sid), &v.kind) {
                (Some(WordValue::Bv { bits, .. }), VarKind::Scalar) => (*bits, mask(v.width)),
                (Some(WordValue::Array { default, cells, .. }), VarKind::Probe { index, .. }) => {
                    (cells.get(index).copied().unwrap_or(*default), mask(v.width))
                }
                (Some(WordValue::Array { default, cells, .. }), VarKind::Lambda { .. }) => {
                    let mut ones = *default;
                    let mut zeros = !*default;
                    for &c in cells.values() {
                        ones &= c;
                        zeros &= !c;
                    }
                    (ones, (ones | zeros) & mask(v.width))
                }
                _ => continue, // nondeterministic at init
            };
            for bit in 0..v.width {
                if (care >> bit) & 1 == 0 {
                    continue;
                }
                let want_one = (val >> bit) & 1 == 1;
                match Self::bit_lit(built, v.in0, bit) {
                    Ok(d) => out.push(Ok(vec![if want_one { d } else { -d }])),
                    Err(c) => out.push(Err(c == want_one)),
                }
            }
        }
        out
    }

    /// Vocabulary cube of the model's frame-0 state (in the given unroller's
    /// input order). Λ vars are EXCLUDED: a model only witnesses one cell
    /// `A[ι_model]`, so blocking its Λ-bits as a ∀-cell clause would
    /// over-claim; Λ literals enter lemmas only via the explicit
    /// (re-verified) Λ-projection in the blocking strategy.
    fn cube_of_model(&self, vals: &[u128], w: UWhich) -> Vec<PLit> {
        let mut cube = Vec::with_capacity(self.atoms.len());
        for v in &self.vars {
            if matches!(v.kind, VarKind::Lambda { .. }) {
                continue;
            }
            let in_idx = match w {
                UWhich::Step => v.in0,
                UWhich::Bad => v.in0_bad,
            };
            let val = vals[in_idx] & mask(v.width);
            for bit in 0..v.width {
                cube.push(plit(v.atom_base + bit as usize, (val >> bit) & 1 == 1));
            }
        }
        cube.sort_unstable();
        cube
    }

    // -- IC3 core ------------------------------------------------------------

    /// Consecution query for cube `s` at level `i`:
    /// `SAT?[ F_{i-1} ∧ ¬s ∧ T ∧ s' ]` (Init asserted when `i == 1`).
    fn consecution(&mut self, i: usize, s: &[PLit]) -> Result<QueryOutcome, String> {
        let q = Query {
            frame_level: Some(i - 1),
            assert_init: i == 1,
            neg_cube0: Some(s),
            cube1: Some(s),
            bad0: false,
        };
        self.solve(&q)
    }

    /// One bounded literal-dropping pass (probe bits first, then state bits,
    /// smallest surviving cube wins); every drop is re-verified by a fresh
    /// consecution query PLUS the exact initiation check.
    fn generalize(&mut self, mut cube: Vec<PLit>, i: usize) -> Result<Vec<PLit>, String> {
        let mut ordered: Vec<PLit> = cube.clone();
        // Probe atoms first (the design's ordering): probes have larger var
        // indices but we want them dropped first; sort by (is_scalar, lit).
        ordered.sort_by_key(|&l| {
            let (vi, _) = self.atoms[plit_atom(l)];
            (matches!(self.vars[vi].kind, VarKind::Scalar), l)
        });
        for l in ordered {
            if cube.len() <= 1 {
                break;
            }
            if self.over_budget("generalization").is_some() {
                break;
            }
            let candidate: Vec<PLit> = cube.iter().copied().filter(|&x| x != l).collect();
            if self.cube_intersects_init(&candidate) {
                continue;
            }
            if matches!(self.consecution(i, &candidate)?, QueryOutcome::Unsat) {
                cube = candidate;
            }
        }
        Ok(cube)
    }

    /// Λ-projections of an obligation cube: for each probe pin present in
    /// `s` whose array carries a Λ pin, one candidate that REPLACES that
    /// probe's literals with the same bit pattern on the array's Λ var
    /// (index generalization before literal generalization). Blocking the
    /// projected cube blocks a SUPERSET of `s`'s states — any state
    /// satisfying `s` witnesses the ∃-cell pattern through the probed cell —
    /// and every candidate is re-verified by consecution + initiation before
    /// it becomes a lemma, so the projection heuristic cannot affect
    /// soundness. Empty whenever `s` carries no probe on a Λ-eligible array
    /// (so on nets with no Λ pin — e.g. simple-stack — this is always empty
    /// and `block_one` degenerates to the phase-3 consecution+generalize).
    fn lambda_projections(&self, s: &[PLit]) -> Vec<Vec<PLit>> {
        // sid -> lambda var index.
        let mut lam_of: HashMap<NodeId, usize> = HashMap::new();
        for (vi, v) in self.vars.iter().enumerate() {
            if matches!(v.kind, VarKind::Lambda { .. }) {
                lam_of.insert(v.sid, vi);
            }
        }
        if lam_of.is_empty() {
            return Vec::new();
        }
        // Probe var -> its literals in s.
        let mut probe_lits: HashMap<usize, Vec<PLit>> = HashMap::new();
        for &l in s {
            let (vi, _) = self.atoms[plit_atom(l)];
            if matches!(self.vars[vi].kind, VarKind::Probe { .. }) {
                probe_lits.entry(vi).or_default().push(l);
            }
        }
        let mut probe_vis: Vec<usize> = probe_lits.keys().copied().collect();
        probe_vis.sort_unstable();
        let mut out = Vec::new();
        for pvi in probe_vis {
            let pv = &self.vars[pvi];
            let Some(&lvi) = lam_of.get(&pv.sid) else {
                continue;
            };
            let lv = &self.vars[lvi];
            let mut cand: Vec<PLit> = Vec::with_capacity(s.len());
            for &l in s {
                let (vi, bit) = self.atoms[plit_atom(l)];
                if vi == pvi {
                    // Same bit pattern, on the Λ atom.
                    cand.push(plit(lv.atom_base + bit as usize, plit_is_one(l)));
                } else {
                    cand.push(l);
                }
            }
            cand.sort_unstable();
            out.push(cand);
        }
        out
    }

    /// Try to block obligation `s` at level `i` via its Λ-projections FIRST
    /// (at most ONE extra consecution query — the design's budget), then via
    /// the const-pin cube itself. `Ok(None)` = blocked (lemma added);
    /// `Ok(Some(model))` = consecution SAT on `s` itself (descend). The
    /// generalize called on either path is the phase-3 LINEAR one (never the
    /// bulk-drop ddmin), so it never produces the under-constrained query
    /// that blows the refinement cap.
    fn block_one(
        &mut self,
        i: usize,
        s: &[PLit],
    ) -> Result<Option<(Vec<u128>, CandidatePlan)>, String> {
        let mut spent_projection_query = false;
        for cand in self.lambda_projections(s) {
            if spent_projection_query {
                break;
            }
            if self.frames.is_blocked(i, &cand) {
                // An existing ∀ lemma subsumes the projection, yet the model
                // reached us (rendered instances are weaker than the ∀ at
                // unprobed cells). Fall through to the const-pin path, which
                // always makes progress (adds a probe-vocabulary lemma or
                // descends) — re-claiming "blocked" here would livelock.
                continue;
            }
            if self.cube_intersects_init(&cand) {
                continue; // free to skip (no query spent)
            }
            spent_projection_query = true;
            if matches!(self.consecution(i, &cand)?, QueryOutcome::Unsat) {
                let g = self.generalize(cand, i)?;
                if self.verbose {
                    eprintln!(
                        "array-ic3: blocked Λ-projected cube at level {i} (lemma {} lits)",
                        g.len()
                    );
                }
                self.frames.add_lemma(i, g);
                return Ok(None);
            }
        }
        match self.consecution(i, s)? {
            QueryOutcome::Unsat => {
                let g = self.generalize(s.to_vec(), i)?;
                if self.verbose {
                    eprintln!(
                        "array-ic3: blocked cube at level {i} (lemma {} lits)",
                        g.len()
                    );
                }
                self.frames.add_lemma(i, g);
                Ok(None)
            }
            QueryOutcome::Model(vals, plan) => Ok(Some((vals, plan))),
        }
    }

    /// Recursively block `cube` at `level` (the standard obligation loop).
    fn block(&mut self, cube: Vec<PLit>, level: usize) -> Result<BlockResult, String> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        // Min-heap on (level, seq): lowest level first, FIFO within a level.
        // Every obligation here descends from the bad seed cube (no Een-style
        // forward re-pushes: those make the loop enumerate predecessors of
        // weakly-generalized lemmas value-by-value — a measured query
        // explosion. Lemma pushing is done wholesale, one solve per lemma per
        // level, in the propagation phase instead).
        let mut heap: BinaryHeap<Reverse<(usize, u64, u64)>> = BinaryHeap::new();
        // seq -> (cube, dist)
        let mut store: HashMap<u64, (Vec<PLit>, usize)> = HashMap::new();

        let seed = self.seq;
        self.seq += 1;
        store.insert(seed, (cube, 0));
        heap.push(Reverse((level, seed, 0)));

        while let Some(Reverse((i, id, _))) = heap.pop() {
            if let Some(r) = self.over_budget("obligation loop") {
                return Err(r);
            }
            let (s, dist) = store.get(&id).cloned().ok_or("obligation store")?;
            if self.frames.is_blocked(i, &s) {
                continue;
            }
            match self.block_one(i, &s)? {
                None => {} // blocked (Λ-projected or const-pin lemma added)
                Some((vals, _plan)) => {
                    let p = self.cube_of_model(&vals, UWhich::Step);
                    if i == 1 || self.cube_intersects_init(&p) {
                        // The bad-rooted chain reached Init: candidate.
                        return Ok(BlockResult::CexCandidate(dist + 1));
                    }
                    let pid = self.seq;
                    self.seq += 1;
                    store.insert(pid, (p, dist + 1));
                    heap.push(Reverse((i - 1, pid, 0)));
                    // Re-enqueue the original obligation.
                    let rid = self.seq;
                    self.seq += 1;
                    store.insert(rid, (s, dist));
                    heap.push(Reverse((i, rid, 0)));
                }
            }
        }
        Ok(BlockResult::Blocked)
    }

    /// The main PDR loop.
    fn run(&mut self, max_frames: usize) -> Result<ArrayIc3Outcome, String> {
        loop {
            let top = self.frames.top();
            if let Some(r) = self.over_budget("frame loop") {
                return Err(r);
            }
            if top > max_frames {
                return Ok(ArrayIc3Outcome::BoundedNoCex {
                    frames_completed: top - 1,
                });
            }

            // Block every bad state in F_top. The bad query runs on the
            // DEPTH-0 bad unroller (`F_top ∧ constr@0 ∧ Bad@0` — the exact
            // triple-S shape): BTOR2 traces may end at the bad frame, so the
            // phase-3 query through the 1-step unroller (which baked
            // `T ∧ constr@1` and thus demanded a legal successor)
            // under-approximated Bad and trivially converged on
            // constraint-dead-end nets.
            loop {
                match self.solve_bad(top)? {
                    QueryOutcome::Unsat => break,
                    QueryOutcome::Model(vals, _plan) => {
                        let s = self.cube_of_model(&vals, UWhich::Bad);
                        if s.is_empty() || self.cube_intersects_init(&s) {
                            return self.confirm_cex(1);
                        }
                        match self.block(s, top)? {
                            BlockResult::Blocked => {}
                            BlockResult::CexCandidate(d) => return self.confirm_cex(d),
                        }
                    }
                }
            }
            if self.verbose {
                eprintln!(
                    "array-ic3: frame {top} bad-free ({} lemmas total)",
                    self.frames.clauses_at(1).count()
                );
            }

            // Extend and propagate.
            self.frames.push_new();
            for i in 1..=top {
                let lemmas: Vec<Lemma> = self.frames.lemmas[i].clone();
                for lemma in lemmas {
                    let q = Query {
                        frame_level: Some(i),
                        assert_init: false,
                        neg_cube0: None,
                        cube1: Some(&lemma.cube),
                        bad0: false,
                    };
                    if matches!(self.solve(&q)?, QueryOutcome::Unsat) {
                        // Push i -> i+1 (delta move).
                        self.frames.lemmas[i].retain(|l| l.cube != lemma.cube);
                        self.frames.add_lemma(i + 1, lemma.cube);
                    }
                }
                if self.frames.lemmas[i].is_empty() {
                    // F_i == F_{i+1}: converged.
                    let inv = self.serialize_invariant(i + 1);
                    if self.verbose {
                        eprintln!(
                            "array-ic3: converged at level {i} ({} clauses, {} probes) — validating triple",
                            inv.clauses.len(),
                            inv.probes.len()
                        );
                    }
                    return Ok(finish_with_validation(
                        self.program,
                        inv,
                        i,
                        self.max_reads,
                        self.verbose,
                    ));
                }
            }
        }
    }

    /// Confirm a candidate counterexample of depth `d` through the existing
    /// replay-gated BMC loop (the identical trust gate as the BMC lane); an
    /// unconfirmed candidate is an honest decline, never a verdict.
    fn confirm_cex(&self, d: usize) -> Result<ArrayIc3Outcome, String> {
        let remaining = self.budget.map(|b| b.saturating_sub(self.start.elapsed()));
        if self.verbose {
            eprintln!("array-ic3: cex candidate at depth ~{d} — confirming via replay-gated BMC");
        }
        let outcome = check_array_bmc(
            self.program,
            &ArrayBmcConfig {
                max_depth: d + 1,
                max_reads: self.max_reads,
                time_budget: remaining,
                verbose: self.verbose,
                ..ArrayBmcConfig::default()
            },
        );
        match outcome {
            ArrayBmcOutcome::Unsafe {
                depth,
                fired,
                model,
                witness,
            } => Ok(ArrayIc3Outcome::Unsafe {
                depth,
                fired,
                model,
                witness,
            }),
            _ => Err(format!(
                "ic3 counterexample candidate at depth ~{d} not confirmed by replay-gated BMC — declined"
            )),
        }
    }

    /// Serialize `F_level` as the standalone invariant artifact.
    fn serialize_invariant(&self, level: usize) -> ArrayFrameInvariant {
        let mut clauses = Vec::new();
        for lemma in self.frames.clauses_at(level) {
            let mut clause = Vec::with_capacity(lemma.cube.len());
            for &l in &lemma.cube {
                let (vi, bit) = self.atoms[plit_atom(l)];
                let v = &self.vars[vi];
                let atom = match &v.kind {
                    VarKind::Scalar => InvAtom::StateBit { state: v.sid, bit },
                    VarKind::Probe { probe_idx, .. } => InvAtom::ProbeBit {
                        probe: *probe_idx,
                        bit,
                    },
                    VarKind::Lambda { lambda_idx } => InvAtom::UCellBit {
                        lambda: *lambda_idx,
                        bit,
                    },
                };
                // Clause literal = negation of the cube literal.
                clause.push(InvLit {
                    atom,
                    positive: !plit_is_one(l),
                });
            }
            clauses.push(clause);
        }
        ArrayFrameInvariant {
            probes: self.probes.clone(),
            lambdas: self.lambdas.clone(),
            clauses,
        }
    }
}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

/// Array states that are read at a SYMBOLIC (non-constant, input-derived)
/// index somewhere in the combinational cone of the bad conditions at frame
/// 0 — the ONLY array states that get a Λ (∀-cell) pin. An array the property
/// does not quantify over (no symbolic read of it in the bad cone — e.g.
/// simple-stack, whose bad cone is two scalar flags) gets none, so its Λ
/// path stays inert and the engine runs the exact phase-3 const-pin lane on
/// it. A pure structural heuristic derived from the net (no magic constant):
/// every ∀-cell lemma it enables is re-verified by the independent LRAT
/// triple gate, so over/under-approximation here only steers work, never
/// trust.
fn lambda_eligible_arrays(program: &Btor2Program) -> HashSet<NodeId> {
    let line_of: HashMap<NodeId, &crate::types::Btor2Line> =
        program.lines.iter().map(|l| (l.id, l)).collect();
    let is_const = |id: NodeId| {
        line_of.get(&id).is_some_and(|l| {
            matches!(
                l.node,
                Btor2Node::Const(_)
                    | Btor2Node::ConstD(_)
                    | Btor2Node::ConstH(_)
                    | Btor2Node::Zero
                    | Btor2Node::One
                    | Btor2Node::Ones
            )
        })
    };
    // Array states reachable from `root` (following write/ite array spines;
    // states are leaves).
    let array_states_under = |root: NodeId, out: &mut HashSet<NodeId>| {
        let mut seen: HashSet<NodeId> = HashSet::new();
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            let Some(line) = line_of.get(&id) else {
                continue;
            };
            if matches!(line.node, Btor2Node::State(_, _)) {
                if matches!(
                    program.sorts.get(&line.sort_id),
                    Some(Btor2Sort::Array { .. })
                ) {
                    out.insert(id);
                }
                continue; // state is a leaf
            }
            stack.extend(line.args.iter().map(|a| a.abs()));
        }
    };
    // Combinational cone of the bad conditions (frame 0: states are leaves).
    let mut cone: HashSet<NodeId> = HashSet::new();
    let mut stack: Vec<NodeId> = Vec::new();
    for &bid in &program.bad_properties {
        if let Some(line) = line_of.get(&bid) {
            stack.extend(line.args.iter().map(|a| a.abs()));
        }
    }
    let mut eligible: HashSet<NodeId> = HashSet::new();
    while let Some(id) = stack.pop() {
        if !cone.insert(id) {
            continue;
        }
        let Some(line) = line_of.get(&id) else {
            continue;
        };
        if matches!(line.node, Btor2Node::State(_, _)) {
            continue; // frame-0 combinational cone: state is a leaf
        }
        if matches!(line.node, Btor2Node::Read) && line.args.len() == 2 {
            let (arr, idx) = (line.args[0].abs(), line.args[1].abs());
            if !is_const(idx) {
                array_states_under(arr, &mut eligible);
            }
        }
        stack.extend(line.args.iter().map(|a| a.abs()));
    }
    eligible
}

fn ic3_inner(program: &Btor2Program, config: &ArrayIc3Config) -> Result<ArrayIc3Outcome, String> {
    lane_supported(program)?;
    let start = Instant::now();

    let init_vals: HashMap<NodeId, WordValue> = eval_init_state_values(program)
        .ok_or("init expression unevaluable in the empty context — declined")?
        .into_iter()
        .collect();

    // Shallow replay-gated BMC (depths 0..=1) first: IC3 obligations assume
    // bad is unreachable in < 2 steps from Init.
    let shallow_budget = config
        .time_budget
        .map(|b| b.min(Duration::from_secs_f64(b.as_secs_f64() * 0.25)));
    match check_array_bmc(
        program,
        &ArrayBmcConfig {
            max_depth: 1,
            max_reads: config.max_reads,
            time_budget: shallow_budget,
            verbose: config.verbose,
            ..ArrayBmcConfig::default()
        },
    ) {
        ArrayBmcOutcome::Unsafe {
            depth,
            fired,
            model,
            witness,
        } => {
            return Ok(ArrayIc3Outcome::Unsafe {
                depth,
                fired,
                model,
                witness,
            })
        }
        ArrayBmcOutcome::BoundedNoCex { .. } => {}
        ArrayBmcOutcome::Declined { reason } => {
            return Err(format!("shallow BMC declined: {reason}"));
        }
    }

    let mut eng = Engine::new(program, config, init_vals, start)?;
    eng.run(config.max_frames)
}

// ---------------------------------------------------------------------------
// Validation triple (independent of the frames engine; LRAT-checked leaves)
// ---------------------------------------------------------------------------

/// Validate + gate: ProvedSafe only when the applicable tier discharges all
/// three checks; anything else downgrades to BoundedNoCex.
fn finish_with_validation(
    program: &Btor2Program,
    inv: ArrayFrameInvariant,
    converged_at: usize,
    read_budget: usize,
    verbose: bool,
) -> ArrayIc3Outcome {
    let tier_a_eligible = flatten_eligible(program);
    let result = if tier_a_eligible && tier_a_expansion_ok(&inv) {
        validate_tier_a(program, &inv).map(|()| ArrayCertTier::FlattenedLrat)
    } else if tier_a_eligible {
        // Tier A resource-declines on the ∀-expansion size (a cross-Λ
        // product): falling to Tier B is a RESOURCE decline, not a trust
        // downgrade — the verdict still requires a full LRAT-checked triple
        // from the one tier that runs.
        validate_tier_b(program, &inv, read_budget).map(|()| ArrayCertTier::EagerOneStepLrat)
    } else {
        validate_tier_b(program, &inv, read_budget).map(|()| ArrayCertTier::EagerOneStepLrat)
    };
    match result {
        Ok(tier) => ArrayIc3Outcome::ProvedSafe {
            converged_at,
            invariant: inv,
            tier,
        },
        Err(why) => {
            if verbose {
                eprintln!(
                    "array-ic3: triple validation FAILED ({why}) — downgrading to bounded-no-cex"
                );
            }
            ArrayIc3Outcome::BoundedNoCex {
                frames_completed: converged_at,
            }
        }
    }
}

/// Tier-A structural eligibility: every array STATE flattens within the
/// certifier's structural caps (mirrors `array_cert`'s bounds — index width
/// and flat bit size; the adaptive gate ceiling is enforced inside the
/// discharge itself).
fn flatten_eligible(program: &Btor2Program) -> bool {
    for line in &program.lines {
        if !matches!(line.node, Btor2Node::State(_, _)) {
            continue;
        }
        let Some(sort) = program.sorts.get(&line.sort_id) else {
            return false;
        };
        if let Some((iw, ew)) = array_dims(sort) {
            let flat = (1u64 << iw.min(63)) * u64::from(ew);
            if iw > 12 || flat > 8192 {
                return false;
            }
        } else if matches!(sort, Btor2Sort::Array { .. }) {
            return false;
        }
    }
    true
}

/// Tier-A ∀-expansion resource gate (derived, no new constants): a clause's
/// ground-instance count is the product of `2^iw` over the DISTINCT Λ pins
/// it references; allow up to the largest single-Λ expansion (single-Λ
/// clauses — the only kind the blocking strategy produces — always fit under
/// the `flatten_eligible` index-width cap; a cross-Λ product declines to
/// Tier B).
fn tier_a_expansion_ok(inv: &ArrayFrameInvariant) -> bool {
    let max_single: u128 = inv
        .lambdas
        .iter()
        .map(|&(_, iw)| 1u128 << iw.min(127))
        .max()
        .unwrap_or(1);
    for clause in &inv.clauses {
        let mut lams: Vec<usize> = clause
            .iter()
            .filter_map(|l| match &l.atom {
                InvAtom::UCellBit { lambda, .. } => Some(*lambda),
                _ => None,
            })
            .collect();
        lams.sort_unstable();
        lams.dedup();
        let mut prod: u128 = 1;
        for lam in lams {
            let iw = inv.lambdas.get(lam).map(|&(_, iw)| iw).unwrap_or(128);
            prod = prod.saturating_mul(1u128 << iw.min(127));
        }
        if prod > max_single {
            return false;
        }
    }
    true
}

/// Tier A: express the invariant as a [`ChcExpr`] over the ORIGINAL
/// program's VC components and discharge the three VCs through
/// `array_cert`'s ground bit-level LRAT route (fully flattened arrays; no
/// epoch/Unroller code in the trust path).
fn validate_tier_a(program: &Btor2Program, inv: &ArrayFrameInvariant) -> Result<(), String> {
    let (_, components) = translate_to_chc_with_vc(program).map_err(|e| format!("to_chc: {e}"))?;

    // Positional params matching the state entries; formula over them.
    let params: Vec<ChcVar> = components
        .state_entries
        .iter()
        .enumerate()
        .map(|(i, e)| ChcVar::new(format!("inv_p{i}"), e.var.sort.clone()))
        .collect();
    let pos_of: HashMap<NodeId, usize> = components
        .state_entries
        .iter()
        .enumerate()
        .map(|(i, e)| (e.node_id, i))
        .collect();

    let lit_expr = |l: &InvLit, lam_at: &HashMap<usize, u128>| -> Result<ChcExpr, String> {
        let (sid, bit, select_idx): (NodeId, u32, Option<u128>) = match &l.atom {
            InvAtom::StateBit { state, bit } => (*state, *bit, None),
            InvAtom::ProbeBit { probe, bit } => {
                let (sid, idx) = inv.probes.get(*probe).ok_or("probe index out of range")?;
                (*sid, *bit, Some(*idx))
            }
            InvAtom::UCellBit { lambda, bit } => {
                let (sid, _) = inv
                    .lambdas
                    .get(*lambda)
                    .ok_or("lambda index out of range")?;
                let idx = lam_at
                    .get(lambda)
                    .copied()
                    .ok_or("unbound ∀-cell index in ground expansion")?;
                (*sid, *bit, Some(idx))
            }
        };
        let &pos = pos_of
            .get(&sid)
            .ok_or("invariant references a non-state node")?;
        let param = &params[pos];
        let base = match select_idx {
            None => ChcExpr::Var(param.clone()),
            Some(idx) => {
                let ChcSort::Array(isort, _) = &param.sort else {
                    return Err("probe on non-array state".into());
                };
                let ChcSort::BitVec(iw) = isort.as_ref() else {
                    return Err("probe on non-bitvector-indexed array".into());
                };
                ChcExpr::Op(
                    ChcOp::Select,
                    vec![
                        Arc::new(ChcExpr::Var(param.clone())),
                        Arc::new(ChcExpr::BitVec(idx & mask(*iw), *iw)),
                    ],
                )
            }
        };
        let bit_e = ChcExpr::Op(ChcOp::BvExtract(bit, bit), vec![Arc::new(base)]);
        Ok(ChcExpr::Op(
            ChcOp::Eq,
            vec![
                Arc::new(bit_e),
                Arc::new(ChcExpr::BitVec(u128::from(l.positive), 1)),
            ],
        ))
    };

    let mut clause_exprs = Vec::with_capacity(inv.clauses.len());
    for clause in &inv.clauses {
        // Distinct Λ pins referenced by this clause: the ∀ prefix to expand.
        let mut lams: Vec<usize> = clause
            .iter()
            .filter_map(|l| match &l.atom {
                InvAtom::UCellBit { lambda, .. } => Some(*lambda),
                _ => None,
            })
            .collect();
        lams.sort_unstable();
        lams.dedup();

        // Odometer over the cartesian product of index domains (the empty
        // prefix yields exactly one ground instance: the clause itself).
        let widths: Vec<u32> = lams
            .iter()
            .map(|&lam| inv.lambdas.get(lam).map(|&(_, iw)| iw).unwrap_or(128))
            .collect();
        if widths.iter().any(|&iw| iw >= 63) {
            return Err("tier A: ∀-cell index too wide to ground-expand".into());
        }
        let mut counter: Vec<u128> = vec![0; lams.len()];
        loop {
            let lam_at: HashMap<usize, u128> =
                lams.iter().copied().zip(counter.iter().copied()).collect();
            let lits: Result<Vec<_>, String> =
                clause.iter().map(|l| lit_expr(l, &lam_at)).collect();
            let lits = lits?;
            clause_exprs.push(match lits.len() {
                0 => ChcExpr::Bool(false),
                1 => lits.into_iter().next().expect("len 1"),
                _ => ChcExpr::Op(ChcOp::Or, lits.into_iter().map(Arc::new).collect()),
            });
            // Advance the odometer; done when it wraps (or is empty).
            let mut pos = 0usize;
            loop {
                if pos == counter.len() {
                    break;
                }
                counter[pos] += 1;
                if counter[pos] < (1u128 << widths[pos]) {
                    break;
                }
                counter[pos] = 0;
                pos += 1;
            }
            if pos == counter.len() {
                break;
            }
        }
    }
    let formula = match clause_exprs.len() {
        0 => ChcExpr::Bool(true),
        1 => clause_exprs.into_iter().next().expect("len 1"),
        _ => ChcExpr::Op(ChcOp::And, clause_exprs.into_iter().map(Arc::new).collect()),
    };

    let invariant = crate::array_cert::Invariant { params, formula };
    match crate::array_cert::discharge_vcs_lrat(&components, &invariant) {
        crate::IndependentCertResult::Certified { .. } => Ok(()),
        crate::IndependentCertResult::NotConfirmed { reason } => Err(format!("tier A: {reason}")),
    }
}

/// How ∀-cell (UCellBit) clauses are rendered in a Tier-B check circuit.
///
/// * `Witness`: each ∀-clause is rendered ONCE at its own fresh free index
///   per Λ (the map key is `(clause index, lambda index)`). EXACT for the
///   NEGATED occurrence: `¬(∀k: C[k])` is satisfiable iff some assignment of
///   the free witness index refutes the instance.
/// * `Premise`: each ∀-clause expands to the conjunction of its instances
///   over a closed-form finite index set per Λ. A sound WEAKENING of the
///   premise: every instance is implied by the true ∀-clause, so the
///   one-directional over-approximation UNSAT argument carries over for ANY
///   instantiation set.
enum LamRender<'m> {
    Witness(&'m HashMap<(usize, usize), i64>),
    Premise(&'m HashMap<usize, Vec<i64>>),
}

/// Term indices of `sid`'s array chain at the given frames (write/ite spine
/// down to the roots).
fn lambda_chain(u: &Unroller<'_>, sid: NodeId, frames: &[usize]) -> HashSet<usize> {
    let mut set = HashSet::new();
    let mut stack: Vec<usize> = frames
        .iter()
        .filter_map(|&f| u.array_state_term(f, sid))
        .collect();
    while let Some(t) = stack.pop() {
        if !set.insert(t) {
            continue;
        }
        match u.terms[t] {
            ATerm::Write { base, .. } => stack.push(base),
            ATerm::Ite { then_t, else_t, .. } => {
                stack.push(then_t);
                stack.push(else_t);
            }
            ATerm::RootInit { .. } | ATerm::RootNondet { .. } => {}
        }
    }
    set
}

/// The closed-form premise instantiation index set for one Λ array: every
/// write address on its chain plus every read index over its chain terms.
fn chain_addr_nodes(u: &Unroller<'_>, chain: &HashSet<usize>) -> Vec<i64> {
    let mut out: Vec<i64> = Vec::new();
    for &t in chain {
        if let ATerm::Write { idx_u, .. } = u.terms[t] {
            out.push(idx_u);
        }
    }
    for r in &u.reads {
        if chain.contains(&r.term) {
            out.push(r.idx_u);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Fresh existential witness reads for every (∀-clause, Λ) pair at `frame`.
/// Returns the witness READ VAR per (clause, lambda) and the witness INDEX
/// NODE per lambda (for inclusion in the premise instantiation set).
fn make_lambda_witnesses(
    u: &mut Unroller<'_>,
    inv: &ArrayFrameInvariant,
    frame: usize,
) -> Result<(HashMap<(usize, usize), i64>, Vec<(usize, i64)>), String> {
    let mut wit: HashMap<(usize, usize), i64> = HashMap::new();
    let mut idx_nodes: Vec<(usize, i64)> = Vec::new();
    for (ci, clause) in inv.clauses.iter().enumerate() {
        let mut lams: Vec<usize> = clause
            .iter()
            .filter_map(|l| match &l.atom {
                InvAtom::UCellBit { lambda, .. } => Some(*lambda),
                _ => None,
            })
            .collect();
        lams.sort_unstable();
        lams.dedup();
        for lam in lams {
            let &(sid, iw) = inv.lambdas.get(lam).ok_or("lambda index out of range")?;
            let t = u
                .array_state_term(frame, sid)
                .ok_or("Λ pin on missing array state")?;
            let (nu, _) = u.fresh_free_index(iw, "wit");
            let r = u.get_or_make_read(t, nu)?;
            wit.insert((ci, lam), u.reads[r].var_u);
            idx_nodes.push((lam, nu));
        }
    }
    Ok((wit, idx_nodes))
}

/// Premise instance reads for every Λ at `frame`: reads of `A@frame` at the
/// closed-form index set (chain write/read addresses over `chain_frames`,
/// plus any extra index nodes — the ¬Inv-side witnesses). Returns the read
/// var list per lambda.
fn make_lambda_premises(
    u: &mut Unroller<'_>,
    inv: &ArrayFrameInvariant,
    frame: usize,
    chain_frames: &[usize],
    extra_idx_nodes: &[(usize, i64)],
) -> Result<HashMap<usize, Vec<i64>>, String> {
    let mut out: HashMap<usize, Vec<i64>> = HashMap::new();
    for (lam, &(sid, _)) in inv.lambdas.iter().enumerate() {
        let chain = lambda_chain(u, sid, chain_frames);
        let mut idxs = chain_addr_nodes(u, &chain);
        for &(l, n) in extra_idx_nodes {
            if l == lam {
                idxs.push(n);
            }
        }
        idxs.sort_unstable();
        idxs.dedup();
        let t = u
            .array_state_term(frame, sid)
            .ok_or("Λ pin on missing array state")?;
        let mut vars = Vec::with_capacity(idxs.len());
        for idx in idxs {
            let r = u.get_or_make_read(t, idx)?;
            vars.push(u.reads[r].var_u);
        }
        out.insert(lam, vars);
    }
    Ok(out)
}

/// Tier B: three fresh one-step encodings with the closed-form EAGER
/// instantiation policy; each check LRAT-discharged. Sound one-directionally:
/// the encodings over-approximate the true one-step semantics, so verified
/// UNSAT discharges the true VC. ∀-cell clauses render per [`LamRender`]:
/// exact fresh-witness reads on every ¬Inv side, finite closed-form
/// instantiation on every premise side.
fn validate_tier_b(
    program: &Btor2Program,
    inv: &ArrayFrameInvariant,
    read_budget: usize,
) -> Result<(), String> {
    // ---- Check I: Init ∧ ¬Inv (real init, depth 0; no step constraints) ----
    {
        let mut u = Unroller::new(program, read_budget, false)?;
        let pins = make_pins(&mut u, inv, 0)?;
        let (wit, _) = make_lambda_witnesses(&mut u, inv, 0)?;
        eager_instantiate(&mut u, read_budget)?;
        let inv0 = emit_inv_circuit(&mut u, inv, &pins, 0, &LamRender::Witness(&wit))?;
        lrat_check(&u, &[-inv0], "I (Init ⊆ Inv)")?;
    }
    // ---- Check C: Inv ∧ T ∧ constraints ∧ ¬Inv' (free init, one step) ------
    {
        let mut u = Unroller::new(program, read_budget, true)?;
        u.seed_through(1)?;
        u.emit_constraints_through(1)?;
        let pins0 = make_pins(&mut u, inv, 0)?;
        let pins1 = make_pins(&mut u, inv, 1)?;
        // ¬Inv' witnesses at frame 1 first, so their index nodes join the
        // frame-0 premise instantiation set (the instance that makes the
        // write-through reasoning close).
        let (wit1, wit_idx) = make_lambda_witnesses(&mut u, inv, 1)?;
        let prem0 = make_lambda_premises(&mut u, inv, 0, &[0, 1], &wit_idx)?;
        eager_instantiate(&mut u, read_budget)?;
        let inv0 = emit_inv_circuit(&mut u, inv, &pins0, 0, &LamRender::Premise(&prem0))?;
        u.emit_constraint(inv0);
        let inv1 = emit_inv_circuit(&mut u, inv, &pins1, 1, &LamRender::Witness(&wit1))?;
        lrat_check(&u, &[-inv1], "C (Inv ∧ T ⊆ Inv')")?;
    }
    // ---- Check S: Inv ∧ constraints ∧ Bad (free init, depth 0) -------------
    {
        let mut u = Unroller::new(program, read_budget, true)?;
        u.emit_constraints_through(0)?;
        let pins = make_pins(&mut u, inv, 0)?;
        let mut bads = Vec::new();
        for i in 0..program.bad_properties.len() {
            let bid = program.bad_properties[i];
            let cond = u.node_cond(bid)?;
            bads.push(u.unroll_ref(0, cond)?);
        }
        // Premise instances AFTER the bad conditions are unrolled, so the
        // bad reads' index nodes are in the closed-form set.
        let prem0 = make_lambda_premises(&mut u, inv, 0, &[0], &[])?;
        eager_instantiate(&mut u, read_budget)?;
        let inv0 = emit_inv_circuit(&mut u, inv, &pins, 0, &LamRender::Premise(&prem0))?;
        u.emit_constraint(inv0);
        lrat_check(&u, &bads, "S (Inv ⊆ ¬Bad)")?;
    }
    Ok(())
}

/// Create the probe pin reads at `frame` for a fresh validation unroller.
/// Returns the read var node per probe (aligned with `inv.probes`).
fn make_pins(
    u: &mut Unroller<'_>,
    inv: &ArrayFrameInvariant,
    frame: usize,
) -> Result<Vec<i64>, String> {
    if frame > 0 {
        u.seed_through(frame)?;
    }
    let mut pins = Vec::with_capacity(inv.probes.len());
    for &(sid, idx) in &inv.probes {
        let t = u
            .array_state_term(frame, sid)
            .ok_or("probe on missing array state")?;
        let (iw, _) = u.terms[t].dims();
        let cidx = u.const_index(iw, idx & mask(iw));
        let r = u.get_or_make_read(t, cidx)?;
        pins.push(u.reads[r].var_u);
    }
    Ok(pins)
}

/// The closed-form eager instantiation policy: every read's FULL structural
/// spine (model-free — both ite branches, every write link, const roots),
/// then ALL root Ackermann congruence pairs per nondet root, then the E2
/// skolem for every array-equality entry. Deterministic, independent of any
/// engine refinement history; adds only semantic-fact axiom instances, so
/// the encoding remains an over-approximation.
fn eager_instantiate(u: &mut Unroller<'_>, read_budget: usize) -> Result<(), String> {
    let mut i = 0usize;
    while i < u.reads.len() {
        if u.reads.len() > read_budget {
            return Err(format!(
                "eager instantiation exceeded the read budget ({read_budget})"
            ));
        }
        instantiate_structural(u, i);
        i += 1;
    }
    // All root congruence pairs (add_root_congruence_pairs dedups).
    let root_reads: Vec<usize> = (0..u.reads.len())
        .filter(|&ri| matches!(u.terms[u.reads[ri].term], ATerm::RootNondet { .. }))
        .collect();
    for ri in root_reads {
        add_root_congruence_pairs(u, ri);
    }
    // E2 skolems for any array-equality entries.
    for ei in 0..u.eqs.len() {
        u.instantiate_skolem(ei);
    }
    Ok(())
}

/// Emit the invariant as a 1-bit circuit over the unroller's frame-`frame`
/// vocabulary (scalar state nodes + pin read vars; ∀-cell clauses per
/// `lam` — see [`LamRender`]). Returns the node id.
fn emit_inv_circuit(
    u: &mut Unroller<'_>,
    inv: &ArrayFrameInvariant,
    pins: &[i64],
    frame: usize,
    lam: &LamRender<'_>,
) -> Result<i64, String> {
    let mut conj: Option<i64> = None;
    let mut push_conj = |u: &mut Unroller<'_>, c: i64| {
        conj = Some(match conj {
            None => c,
            Some(acc) => u.emit(Btor2Node::And, 1, vec![acc, c]),
        });
    };
    for (ci, clause) in inv.clauses.iter().enumerate() {
        let mut lams: Vec<usize> = clause
            .iter()
            .filter_map(|l| match &l.atom {
                InvAtom::UCellBit { lambda, .. } => Some(*lambda),
                _ => None,
            })
            .collect();
        lams.sort_unstable();
        lams.dedup();

        // One clause instance under a (lambda -> read var) assignment.
        let instance =
            |u: &mut Unroller<'_>, lam_var: &HashMap<usize, i64>| -> Result<i64, String> {
                let mut disj: Option<i64> = None;
                for lit in clause {
                    let (node, bit) = match &lit.atom {
                        InvAtom::StateBit { state, bit } => {
                            let n = u
                                .scalar_state_at(frame, *state)
                                .ok_or("invariant references unseeded scalar state")?;
                            (n, *bit)
                        }
                        InvAtom::ProbeBit { probe, bit } => {
                            let n = *pins.get(*probe).ok_or("probe index out of range")?;
                            (n, *bit)
                        }
                        InvAtom::UCellBit { lambda, bit } => {
                            let n = *lam_var
                                .get(lambda)
                                .ok_or("unbound Λ read var in clause instance")?;
                            (n, *bit)
                        }
                    };
                    let b = u.emit(Btor2Node::Slice(bit, bit), 1, vec![node]);
                    let l = if lit.positive { b } else { -b };
                    disj = Some(match disj {
                        None => l,
                        Some(acc) => u.emit(Btor2Node::Or, 1, vec![acc, l]),
                    });
                }
                disj.ok_or_else(|| "empty invariant clause".to_string())
            };

        if lams.is_empty() {
            let c = instance(u, &HashMap::new())?;
            push_conj(u, c);
            continue;
        }
        match lam {
            LamRender::Witness(wit) => {
                // Exact ∃-witness rendering for the negated occurrence.
                let mut lam_var = HashMap::new();
                for &l in &lams {
                    let v = *wit
                        .get(&(ci, l))
                        .ok_or("missing Λ witness read for clause")?;
                    lam_var.insert(l, v);
                }
                let c = instance(u, &lam_var)?;
                push_conj(u, c);
            }
            LamRender::Premise(prem) => {
                // Finite instantiation (sound premise weakening): the
                // conjunction over the cartesian product of each Λ's
                // closed-form read list. An empty list contributes no
                // instances (the clause weakens to `true`).
                let lists: Vec<&Vec<i64>> = lams
                    .iter()
                    .map(|l| prem.get(l).ok_or("missing Λ premise reads"))
                    .collect::<Result<_, _>>()?;
                if lists.iter().any(|v| v.is_empty()) {
                    continue;
                }
                let mut counter: Vec<usize> = vec![0; lams.len()];
                loop {
                    let lam_var: HashMap<usize, i64> = lams
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(k, l)| (l, lists[k][counter[k]]))
                        .collect();
                    let c = instance(u, &lam_var)?;
                    push_conj(u, c);
                    let mut pos = 0usize;
                    loop {
                        if pos == counter.len() {
                            break;
                        }
                        counter[pos] += 1;
                        if counter[pos] < lists[pos].len() {
                            break;
                        }
                        counter[pos] = 0;
                        pos += 1;
                    }
                    if pos == counter.len() {
                        break;
                    }
                }
            }
        }
    }
    match conj {
        Some(c) => Ok(c),
        // Empty invariant = true.
        None => Ok(u.emit(Btor2Node::One, 1, vec![])),
    }
}

/// Assemble, bit-blast, and discharge one validation check through the
/// LRAT-verified leaf. Any non-verified outcome is an error (downgrade).
fn lrat_check(u: &Unroller<'_>, bad_conds: &[i64], which: &str) -> Result<(), String> {
    let program = u.assemble_program(bad_conds);
    let circuit: BitblastedCircuit =
        bitblast(&program, 128).map_err(|e| format!("{which}: bit-blast: {e}"))?;
    discharge_unsat_lrat(&circuit).map_err(|e| format!("{which}: {e}"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn run(net: &str) -> ArrayIc3Outcome {
        let prog = parse(net).expect("parse");
        check_array_ic3(
            &prog,
            &ArrayIc3Config {
                verbose: std::env::var("TY_ARRAY_IC3_TEST_VERBOSE").is_ok(),
                ..ArrayIc3Config::default()
            },
        )
    }

    /// Wide (iw=16, bit-blast-ineligible) SAFE net: mem init 0, the chain
    /// writes 5 at index 0 every step, bad hunts for 6. The frames must
    /// converge and the triple must discharge through Tier B (flattening is
    /// structurally impossible at iw=16).
    #[test]
    fn ic3_proves_wide_safe_net_tier_b() {
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
        match run(net) {
            ArrayIc3Outcome::ProvedSafe {
                tier, invariant, ..
            } => {
                assert_eq!(tier, ArrayCertTier::EagerOneStepLrat);
                assert!(
                    !invariant.clauses.is_empty(),
                    "invariant must be nontrivial"
                );
            }
            other => panic!("expected ProvedSafe(Tier B), got {other:?}"),
        }
    }

    /// Narrow twin (iw=4): structurally flattenable, so Tier A (the ground
    /// bit-level array_cert route) is MANDATORY and must discharge.
    #[test]
    fn ic3_proves_narrow_safe_net_tier_a() {
        let net = "\
1 sort bitvec 4
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
        match run(net) {
            ArrayIc3Outcome::ProvedSafe { tier, .. } => {
                assert_eq!(tier, ArrayCertTier::FlattenedLrat);
            }
            other => panic!("expected ProvedSafe(Tier A), got {other:?}"),
        }
    }

    /// NOT 1-inductive SAFE net (the class k-induction needs deep k for and
    /// IC3 strengthens through): mem[0] cycles 0 -> 1 -> 2 -> 0; bad = 5.
    /// The 1-step consecution of "mem[0] != 5" alone is SAT (m0 = 4 steps to
    /// 5), so frames must strengthen with reachability lemmas.
    #[test]
    fn ic3_proves_non_1inductive_cycle_net() {
        let net = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
8 zero 1
9 read 2 4 8
12 sort bitvec 1
20 constd 2 2
21 eq 12 9 20
22 constd 2 0
23 constd 2 1
24 add 2 9 23
25 ite 2 21 22 24
26 write 3 4 8 25
7 next 3 4 26
27 constd 2 5
28 eq 12 9 27
29 bad 28
";
        match run(net) {
            ArrayIc3Outcome::ProvedSafe { invariant, .. } => {
                assert!(!invariant.clauses.is_empty());
            }
            other => panic!("expected ProvedSafe, got {other:?}"),
        }
    }

    /// UNSAFE at depth 1 (write 5 then read it): the lane must confirm via
    /// the replay-gated BMC path — never an unreplayed verdict.
    #[test]
    fn ic3_unsafe_is_replay_gated() {
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
        match run(net) {
            ArrayIc3Outcome::Unsafe { depth, fired, .. } => {
                assert_eq!(depth, 1);
                assert_eq!(fired, vec![0]);
            }
            other => panic!("expected Unsafe@1, got {other:?}"),
        }
    }

    /// The phase-3 two-chains-one-cell stall fixture (see array_bmc tests):
    /// IC3 must also prove it (the property is a pure array-semantics fact).
    #[test]
    fn ic3_proves_two_chains_one_cell() {
        let net = "\
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
        match run(net) {
            ArrayIc3Outcome::ProvedSafe { .. } => {}
            // An honest bounded outcome is acceptable (the property is
            // combinational, so frames may converge trivially), but any
            // decline/unsafe is a bug.
            ArrayIc3Outcome::BoundedNoCex { .. } => {}
            other => panic!("expected ProvedSafe/BoundedNoCex, got {other:?}"),
        }
    }

    /// The constraint-dead-end trivial-convergence class (phase-4 fix A):
    /// SAFE net where every `mem[0]==15` state satisfies `constr@0` but has
    /// NO legal successor (`c' = mem[0] = 15` violates `constr@1`). The
    /// phase-3 bad query (through the 1-step unroller, `T ∧ constr@0..1`
    /// baked) was UNSAT at every level -> zero lemmas -> junk `Inv = true`
    /// -> the triple gate rejected it (BoundedNoCex). The frame-0-only bad
    /// query must make the frames do real work and mint a validated proof.
    #[test]
    fn ic3_proves_constraint_dead_end_net() {
        let net = "\
1 sort bitvec 4
2 sort array 1 1
3 sort bitvec 1
4 zero 1
5 state 2 mem
6 init 2 5 4
7 next 2 5 5
8 state 1 c
9 init 1 8 4
10 read 1 5 4
11 next 1 8 10
12 constd 1 15
13 eq 3 8 12
14 not 3 13
15 constraint 14
16 eq 3 10 12
17 bad 16
";
        match run(net) {
            ArrayIc3Outcome::ProvedSafe { tier, .. } => {
                assert_eq!(tier, ArrayCertTier::FlattenedLrat);
            }
            other => panic!("expected ProvedSafe(Tier A), got {other:?}"),
        }
    }

    /// The Λ-pin universal-cell vocabulary (phase-4 fix C), miniature
    /// array_lt200: every write is masked below 8, the bad hunts for
    /// `mem[raddr] >= 8` at a FREE read address. Constant-index pins cannot
    /// express the needed `forall i: mem[i] < 8`, so a proof REQUIRES the
    /// ∀-cell vocabulary (and its Tier-A ground expansion).
    #[test]
    fn ic3_proves_universal_cell_bound_net() {
        let net = "\
1 sort bitvec 4
2 sort bitvec 8
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
7 input 1 waddr
8 input 2 wdata
9 constd 2 7
10 and 2 8 9
11 write 3 4 7 10
12 next 3 4 11
13 input 1 raddr
14 read 2 4 13
15 sort bitvec 1
16 constd 2 8
17 ugte 15 14 16
18 bad 17
";
        match run(net) {
            ArrayIc3Outcome::ProvedSafe {
                tier, invariant, ..
            } => {
                assert_eq!(tier, ArrayCertTier::FlattenedLrat);
                assert!(
                    invariant
                        .clauses
                        .iter()
                        .any(|c| c.iter().any(|l| matches!(l.atom, InvAtom::UCellBit { .. }))),
                    "the proof must use the ∀-cell vocabulary: {invariant:?}"
                );
            }
            other => panic!("expected ProvedSafe(Tier A) via Λ pins, got {other:?}"),
        }
    }

    /// The triple gate must REJECT deliberately-wrong invariants — through
    /// BOTH tiers, including wrong ∀-cell claims. This is the soundness
    /// envelope for every engine change: generalization/vocabulary bugs can
    /// only propose junk that dies here.
    #[test]
    fn triple_gate_rejects_wrong_invariants() {
        // The narrow safe net from `ic3_proves_narrow_safe_net_tier_a`.
        let net = "\
1 sort bitvec 4
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
        let prog = parse(net).expect("parse");
        // Junk 1: Inv = true (the dead-end signature). Check S must be SAT.
        let junk_true = ArrayFrameInvariant {
            probes: vec![],
            lambdas: vec![],
            clauses: vec![],
        };
        assert!(
            validate_tier_a(&prog, &junk_true).is_err(),
            "tier A must reject Inv = true"
        );
        assert!(
            validate_tier_b(&prog, &junk_true, 512).is_err(),
            "tier B must reject Inv = true"
        );
        // Junk 2: a wrong ∀-cell claim (`forall i: mem[i] bit0 == 1` while
        // init is all-zero). Check I must be SAT.
        let junk_forall = ArrayFrameInvariant {
            probes: vec![],
            lambdas: vec![(4, 4)],
            clauses: vec![vec![InvLit {
                atom: InvAtom::UCellBit { lambda: 0, bit: 0 },
                positive: true,
            }]],
        };
        assert!(
            validate_tier_a(&prog, &junk_forall).is_err(),
            "tier A must reject the wrong ∀-cell claim"
        );
        assert!(
            validate_tier_b(&prog, &junk_forall, 512).is_err(),
            "tier B must reject the wrong ∀-cell claim"
        );
    }

    /// Scalar-only nets stay outside the lane (shared preflight).
    #[test]
    fn ic3_declines_scalar_only_net() {
        let net = "\
1 sort bitvec 4
2 state 1 c
3 zero 1
4 init 1 2 3
5 one 1
6 add 1 2 5
7 next 1 2 6
8 sort bitvec 1
9 constd 1 15
10 eq 8 2 9
11 bad 10
";
        match run(net) {
            ArrayIc3Outcome::Declined { reason } => {
                assert!(reason.contains("no array state"), "reason: {reason}");
            }
            other => panic!("expected Declined, got {other:?}"),
        }
    }
}
