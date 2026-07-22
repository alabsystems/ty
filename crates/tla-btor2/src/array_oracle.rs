// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! K-step explicit-state BFS oracle for tiny BTOR2 array nets.
//!
//! **TEST/DIFFERENTIAL-ONLY** — this module is compiled only under `cfg(test)`
//! and never participates in a production verdict path. It exists because the
//! previous ground-truth harness (`bitblast::bad_reachable`) is a SINGLE-STEP
//! combinational oracle: it enumerates one frame with free latches, so it is
//! provably blind to across-step array write chains (a net that is UNSAFE only
//! via a write at step `t` read at step `t+k`, and a net that is SAFE because
//! init/next constrain that chain, can be indistinguishable to it — see the
//! `single_step_oracle_blindness` regression in `array_battery`). This module
//! closes that gap: an exact bounded-depth enumerator whose verdicts are
//! trusted by *simplicity* (a direct interpreter + exhaustive BFS, no solver).
//!
//! # Semantics
//!
//! A state point is the ordered tuple over the net's `state` lines: scalars as
//! width-masked `u128`s, arrays as canonical concrete maps ([`OArr`]). Frame 0
//! evaluates `init` lines concretely (const-only expressions; a scalar init of
//! an array state broadcasts to a const-array, mirroring `bitblast` and
//! `word_replay::lift_init_value`); states *without* `init` are enumerated
//! exhaustively. Each step enumerates ALL input assignments, prunes branches
//! whose `constraint`s are false (assume semantics, matching
//! `word_replay::replay_collect_bad`), fires on any `bad` that holds with
//! constraints satisfied, and otherwise commits all `next` values
//! simultaneously against the current frame. BFS over a canonical-state
//! visited set up to depth K; full exploration with no bad is an exact
//! K-bounded-safe verdict.
//!
//! # Independence and cross-checking
//!
//! The evaluator here is deliberately NOT a reuse of `word_replay`'s
//! `Evaluator` — sharing it would correlate the oracle with the very replay
//! validator the lanes under test use. Instead, every oracle-found trace is
//! converted to a [`WordLevelModel`] and re-replayed through
//! `word_replay::replay_collect_bad`; the two independent implementations must
//! agree that the trace reaches the same bad properties (the harness panics —
//! build-failing — otherwise).
//!
//! # Decline rules (fail-closed)
//!
//! Any op outside the whitelist, any `state` with NO `next` line (the
//! havoc-vs-hold cross-lane divergence documented in
//! `docs/hwmcc/no-next-havoc-vs-hold-divergence.md` — the oracle declines
//! rather than picking a side), array-sorted inputs, non-bitvector array
//! index/element sorts, non-constant `init` expressions, or any size over the
//! caps returns [`OracleOutcome::Declined`]. Per the differential harness's
//! absolute rule, a declined case is REMOVED from the corpus, never trusted.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use crate::types::{Btor2Line, Btor2Node, Btor2Program, Btor2Sort, NodeId};
use crate::word_replay::{InitialState, InputFrame, WordLevelModel, WordValue};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Size/effort caps for the oracle. The defaults suffice because the bug
/// classes the differential harness hunts (missed congruence pairs, across-step
/// write chains, init/const-array defaults, ite-of-array distribution) are
/// logical/structural and all manifest on 1-2 arrays with 4-8 cells over 2-6
/// steps — scale adds cost, not coverage (the same argument behind
/// `bad_reachable`'s 22-bit cap, lifted from one step to K).
pub(crate) struct OracleConfig {
    /// Max depth K (number of transitions). Clamped to 64.
    pub max_depth: usize,
    /// Cap on the total nondeterministic frame-0 bits (no-`init` states;
    /// an array costs `2^iw * ew` bits).
    pub max_frame0_nondet_bits: u64,
    /// Cap on the total input bits enumerated per step.
    pub max_input_bits_per_step: u64,
    /// Approximate byte budget for the visited set / BFS arena.
    pub visited_byte_budget: usize,
}

impl Default for OracleConfig {
    fn default() -> Self {
        OracleConfig {
            max_depth: 64,
            max_frame0_nondet_bits: 16,
            max_input_bits_per_step: 8,
            visited_byte_budget: 64 << 20, // 64 MiB — test-only, OOM-disciplined
        }
    }
}

/// Hard cap on visited states independent of the byte budget.
const MAX_VISITED_STATES: usize = 1 << 20;

// ---------------------------------------------------------------------------
// Outcome
// ---------------------------------------------------------------------------

/// Oracle verdict for one net at depth K.
#[derive(Debug)]
pub(crate) enum OracleOutcome {
    /// A bad property genuinely fires within K steps. `model` replays through
    /// `word_replay` (asserted before this is returned).
    Unsafe {
        /// Depth (frame index) at which the first bad fires — minimal, since
        /// the exploration is breadth-first.
        depth: usize,
        /// Indices into `program.bad_properties` that fire at that frame.
        fired: Vec<usize>,
        /// The concrete trace as a word-level model (frame-0 nondet states +
        /// per-frame inputs), cross-validated via `word_replay`.
        model: WordLevelModel,
    },
    /// No bad state is reachable within `explored_depth` transitions. Exact
    /// (every reachable state within the bound was enumerated). If `exhausted`
    /// is true the frontier emptied before the depth bound — the reachable
    /// state space is closed, so the net is safe at EVERY depth.
    BoundedSafe {
        /// The depth bound that was fully explored.
        explored_depth: usize,
        /// True iff exploration reached a fixpoint before the bound.
        exhausted: bool,
    },
    /// The net is outside the oracle's trusted class. Never a verdict — a
    /// declined case must be dropped from the differential corpus.
    Declined(String),
}

// ---------------------------------------------------------------------------
// Canonical concrete values
// ---------------------------------------------------------------------------

/// A canonical concrete array: `(index_width, elem_width, default, cells)` with
/// every cell equal to `default` removed and `default` chosen as the value
/// covering the most indices (ties broken toward the smallest value), which
/// makes the representation UNIQUE per abstract array — dedup is exact.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct OArr {
    iw: u32,
    ew: u32,
    default: u128,
    cells: BTreeMap<u128, u128>,
}

impl OArr {
    fn read(&self, idx: u128) -> u128 {
        let key = idx & mask(self.iw);
        self.cells.get(&key).copied().unwrap_or(self.default)
    }

    fn write(&self, idx: u128, val: u128) -> OArr {
        let mut out = self.clone();
        out.cells.insert(idx & mask(self.iw), val & mask(self.ew));
        out.canonicalize();
        out
    }

    /// Restore the unique canonical form: drop default-valued cells, and if
    /// another value now covers more indices than `default` (possible only on
    /// tiny domains where explicit cells dominate), re-root the map on it.
    fn canonicalize(&mut self) {
        self.cells.retain(|_, v| *v != self.default);
        let domain: u128 = if self.iw >= 128 {
            u128::MAX
        } else {
            1u128 << self.iw
        };
        let default_count = domain - self.cells.len() as u128;
        // Count explicit values.
        let mut counts: BTreeMap<u128, u128> = BTreeMap::new();
        for &v in self.cells.values() {
            *counts.entry(v).or_insert(0) += 1;
        }
        // Candidate: (count, prefer-smaller-value). Default wins ties against
        // larger values and against equal-count larger values; a strictly
        // greater count, or an equal count with a smaller value, re-roots.
        let mut best_val = self.default;
        let mut best_count = default_count;
        for (&v, &c) in &counts {
            if c > best_count || (c == best_count && v < best_val) {
                best_val = v;
                best_count = c;
            }
        }
        if best_val != self.default {
            // Rebuild with the new default: indices NOT in cells had the old
            // default; cells with the new default vanish.
            let old_default = self.default;
            let old_cells = std::mem::take(&mut self.cells);
            // Every index of the (small) domain not in old_cells now becomes
            // an explicit cell holding old_default. This branch can only be
            // reached when explicit cells cover >= half the domain, so the
            // domain is small (<= 2 * cells).
            let mut new_cells: BTreeMap<u128, u128> = BTreeMap::new();
            for i in 0..domain {
                let v = old_cells.get(&i).copied().unwrap_or(old_default);
                if v != best_val {
                    new_cells.insert(i, v);
                }
            }
            self.default = best_val;
            self.cells = new_cells;
        }
    }
}

/// A canonical concrete value: width-masked scalar or canonical array.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum OVal {
    /// Bitvector, masked to `width`.
    Bv { bits: u128, width: u32 },
    /// Canonical concrete array.
    Arr(OArr),
}

impl OVal {
    fn as_bv(&self) -> Result<(u128, u32), String> {
        match self {
            OVal::Bv { bits, width } => Ok((*bits, *width)),
            OVal::Arr(_) => Err("expected bitvector, got array".into()),
        }
    }

    fn approx_bytes(&self) -> usize {
        match self {
            OVal::Bv { .. } => 32,
            OVal::Arr(a) => 64 + a.cells.len() * 48,
        }
    }
}

fn mask(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

fn to_signed(bits: u128, width: u32) -> i128 {
    let m = bits & mask(width);
    if width > 0 && width < 128 && (m >> (width - 1)) & 1 == 1 {
        (m as i128) - (1i128 << width)
    } else {
        m as i128
    }
}

// ---------------------------------------------------------------------------
// Independent whitelisted evaluator
// ---------------------------------------------------------------------------

struct OracleEval<'a> {
    program: &'a Btor2Program,
    line_index: &'a HashMap<NodeId, &'a Btor2Line>,
    state: &'a HashMap<NodeId, OVal>,
    inputs: &'a HashMap<NodeId, u128>,
    memo: HashMap<NodeId, OVal>,
}

impl<'a> OracleEval<'a> {
    fn new(
        program: &'a Btor2Program,
        line_index: &'a HashMap<NodeId, &'a Btor2Line>,
        state: &'a HashMap<NodeId, OVal>,
        inputs: &'a HashMap<NodeId, u128>,
    ) -> Self {
        OracleEval {
            program,
            line_index,
            state,
            inputs,
            memo: HashMap::new(),
        }
    }

    fn width_of(&self, line: &Btor2Line) -> Result<u32, String> {
        match self.program.sorts.get(&line.sort_id) {
            Some(Btor2Sort::BitVec(w)) => Ok(*w),
            Some(Btor2Sort::Array { .. }) => Err("expected scalar sort, got array".into()),
            None => Err(format!(
                "missing sort {} for node {}",
                line.sort_id, line.id
            )),
        }
    }

    /// Evaluate an operand, honoring the negative-id = bitwise-NOT shorthand.
    fn eval(&mut self, id: NodeId) -> Result<OVal, String> {
        let neg = id < 0;
        let abs = id.unsigned_abs() as i64;
        let base = self.eval_abs(abs)?;
        if !neg {
            return Ok(base);
        }
        match base {
            OVal::Bv { bits, width } => Ok(OVal::Bv {
                bits: (!bits) & mask(width),
                width,
            }),
            OVal::Arr(_) => Err("negated array operand".into()),
        }
    }

    fn eval_bv(&mut self, id: NodeId) -> Result<(u128, u32), String> {
        self.eval(id)?.as_bv()
    }

    fn eval_abs(&mut self, abs: NodeId) -> Result<OVal, String> {
        if let Some(v) = self.memo.get(&abs) {
            return Ok(v.clone());
        }
        let v = self.eval_uncached(abs)?;
        self.memo.insert(abs, v.clone());
        Ok(v)
    }

    fn arg(&self, line: &Btor2Line, i: usize) -> Result<NodeId, String> {
        line.args
            .get(i)
            .copied()
            .ok_or_else(|| format!("node {} missing arg {i}", line.id))
    }

    fn bin(
        &mut self,
        line: &Btor2Line,
        f: impl Fn(u128, u128, u32) -> u128,
    ) -> Result<OVal, String> {
        let (a, w) = self.eval_bv(self.arg(line, 0)?)?;
        let (b, _) = self.eval_bv(self.arg(line, 1)?)?;
        Ok(OVal::Bv {
            bits: f(a, b, w) & mask(w),
            width: w,
        })
    }

    fn cmp(
        &mut self,
        line: &Btor2Line,
        f: impl Fn(u128, u128, u32) -> bool,
    ) -> Result<OVal, String> {
        let (a, w) = self.eval_bv(self.arg(line, 0)?)?;
        let (b, _) = self.eval_bv(self.arg(line, 1)?)?;
        Ok(OVal::Bv {
            bits: u128::from(f(a & mask(w), b & mask(w), w)),
            width: 1,
        })
    }

    fn eval_uncached(&mut self, abs: NodeId) -> Result<OVal, String> {
        let line = *self
            .line_index
            .get(&abs)
            .ok_or_else(|| format!("undefined node {abs}"))?;
        match &line.node {
            Btor2Node::State(_, _) => self
                .state
                .get(&abs)
                .cloned()
                .ok_or_else(|| format!("state {abs} has no current value")),
            Btor2Node::Input(sort_id, _) => {
                let w = match self.program.sorts.get(sort_id) {
                    Some(Btor2Sort::BitVec(w)) => *w,
                    _ => return Err("array-sorted input (outside oracle class)".into()),
                };
                let v = self
                    .inputs
                    .get(&abs)
                    .copied()
                    .ok_or_else(|| format!("input {abs} not supplied"))?;
                Ok(OVal::Bv {
                    bits: v & mask(w),
                    width: w,
                })
            }

            // -- Constants -----------------------------------------------------
            Btor2Node::Zero => Ok(OVal::Bv {
                bits: 0,
                width: self.width_of(line)?,
            }),
            Btor2Node::One => {
                let w = self.width_of(line)?;
                Ok(OVal::Bv {
                    bits: 1 & mask(w),
                    width: w,
                })
            }
            Btor2Node::Ones => {
                let w = self.width_of(line)?;
                Ok(OVal::Bv {
                    bits: mask(w),
                    width: w,
                })
            }
            Btor2Node::Const(s) => {
                let w = self.width_of(line)?;
                let v = u128::from_str_radix(s, 2).map_err(|e| format!("const: {e}"))?;
                Ok(OVal::Bv {
                    bits: v & mask(w),
                    width: w,
                })
            }
            Btor2Node::ConstD(s) => {
                let w = self.width_of(line)?;
                let v = if let Some(stripped) = s.strip_prefix('-') {
                    let a: u128 = stripped.parse().map_err(|e| format!("constd: {e}"))?;
                    a.wrapping_neg()
                } else {
                    s.parse::<u128>().map_err(|e| format!("constd: {e}"))?
                };
                Ok(OVal::Bv {
                    bits: v & mask(w),
                    width: w,
                })
            }
            Btor2Node::ConstH(s) => {
                let w = self.width_of(line)?;
                let v = u128::from_str_radix(s, 16).map_err(|e| format!("consth: {e}"))?;
                Ok(OVal::Bv {
                    bits: v & mask(w),
                    width: w,
                })
            }

            // -- Unary -----------------------------------------------------------
            Btor2Node::Not => {
                let (a, w) = self.eval_bv(self.arg(line, 0)?)?;
                Ok(OVal::Bv {
                    bits: (!a) & mask(w),
                    width: w,
                })
            }
            Btor2Node::Neg => {
                let (a, w) = self.eval_bv(self.arg(line, 0)?)?;
                Ok(OVal::Bv {
                    bits: a.wrapping_neg() & mask(w),
                    width: w,
                })
            }
            Btor2Node::Inc => {
                let (a, w) = self.eval_bv(self.arg(line, 0)?)?;
                Ok(OVal::Bv {
                    bits: a.wrapping_add(1) & mask(w),
                    width: w,
                })
            }
            Btor2Node::Dec => {
                let (a, w) = self.eval_bv(self.arg(line, 0)?)?;
                Ok(OVal::Bv {
                    bits: a.wrapping_sub(1) & mask(w),
                    width: w,
                })
            }
            Btor2Node::Redand => {
                let (a, w) = self.eval_bv(self.arg(line, 0)?)?;
                Ok(OVal::Bv {
                    bits: u128::from(a & mask(w) == mask(w)),
                    width: 1,
                })
            }
            Btor2Node::Redor => {
                let (a, w) = self.eval_bv(self.arg(line, 0)?)?;
                Ok(OVal::Bv {
                    bits: u128::from(a & mask(w) != 0),
                    width: 1,
                })
            }
            Btor2Node::Redxor => {
                let (a, w) = self.eval_bv(self.arg(line, 0)?)?;
                Ok(OVal::Bv {
                    bits: u128::from((a & mask(w)).count_ones() & 1 == 1),
                    width: 1,
                })
            }

            // -- Binary ----------------------------------------------------------
            Btor2Node::Add => self.bin(line, |a, b, _| a.wrapping_add(b)),
            Btor2Node::Sub => self.bin(line, |a, b, _| a.wrapping_sub(b)),
            Btor2Node::And => self.bin(line, |a, b, _| a & b),
            Btor2Node::Or => self.bin(line, |a, b, _| a | b),
            Btor2Node::Xor => self.bin(line, |a, b, _| a ^ b),
            Btor2Node::Nand => self.bin(line, |a, b, _| !(a & b)),
            Btor2Node::Nor => self.bin(line, |a, b, _| !(a | b)),
            Btor2Node::Xnor => self.bin(line, |a, b, _| !(a ^ b)),
            Btor2Node::Ult => self.cmp(line, |a, b, _| a < b),
            Btor2Node::Ulte => self.cmp(line, |a, b, _| a <= b),
            Btor2Node::Ugt => self.cmp(line, |a, b, _| a > b),
            Btor2Node::Ugte => self.cmp(line, |a, b, _| a >= b),
            Btor2Node::Slt => self.cmp(line, |a, b, w| to_signed(a, w) < to_signed(b, w)),
            Btor2Node::Slte => self.cmp(line, |a, b, w| to_signed(a, w) <= to_signed(b, w)),
            Btor2Node::Sgt => self.cmp(line, |a, b, w| to_signed(a, w) > to_signed(b, w)),
            Btor2Node::Sgte => self.cmp(line, |a, b, w| to_signed(a, w) >= to_signed(b, w)),
            Btor2Node::Iff => {
                let (a, _) = self.eval_bv(self.arg(line, 0)?)?;
                let (b, _) = self.eval_bv(self.arg(line, 1)?)?;
                Ok(OVal::Bv {
                    bits: u128::from((a & 1) == (b & 1)),
                    width: 1,
                })
            }
            Btor2Node::Implies => {
                let (a, _) = self.eval_bv(self.arg(line, 0)?)?;
                let (b, _) = self.eval_bv(self.arg(line, 1)?)?;
                Ok(OVal::Bv {
                    bits: u128::from((a & 1) == 0 || (b & 1) == 1),
                    width: 1,
                })
            }

            // -- Equality (scalar or extensional array) -------------------------
            Btor2Node::Eq | Btor2Node::Neq => {
                let a = self.eval(self.arg(line, 0)?)?;
                let b = self.eval(self.arg(line, 1)?)?;
                let eq = match (&a, &b) {
                    (OVal::Bv { bits: x, width: w }, OVal::Bv { bits: y, .. }) => {
                        (x & mask(*w)) == (y & mask(*w))
                    }
                    // Canonical form is unique per abstract array, so
                    // extensional equality IS structural equality.
                    (OVal::Arr(x), OVal::Arr(y)) => x == y,
                    _ => return Err("eq between array and scalar".into()),
                };
                let bit = if matches!(line.node, Btor2Node::Eq) {
                    eq
                } else {
                    !eq
                };
                Ok(OVal::Bv {
                    bits: u128::from(bit),
                    width: 1,
                })
            }

            // -- Structure -------------------------------------------------------
            Btor2Node::Concat => {
                let (a, _) = self.eval_bv(self.arg(line, 0)?)?;
                let (b, wb) = self.eval_bv(self.arg(line, 1)?)?;
                let w = self.width_of(line)?;
                Ok(OVal::Bv {
                    bits: ((a << wb) | (b & mask(wb))) & mask(w),
                    width: w,
                })
            }
            Btor2Node::Slice(upper, lower) => {
                let (a, _) = self.eval_bv(self.arg(line, 0)?)?;
                let w = upper - lower + 1;
                Ok(OVal::Bv {
                    bits: (a >> lower) & mask(w),
                    width: w,
                })
            }
            Btor2Node::Uext(n) => {
                let (a, aw) = self.eval_bv(self.arg(line, 0)?)?;
                Ok(OVal::Bv {
                    bits: a & mask(aw),
                    width: aw + n,
                })
            }
            Btor2Node::Sext(n) => {
                let (a, aw) = self.eval_bv(self.arg(line, 0)?)?;
                let w = aw + n;
                let s = to_signed(a, aw);
                Ok(OVal::Bv {
                    bits: (s as u128) & mask(w),
                    width: w,
                })
            }

            // -- Control / arrays ------------------------------------------------
            Btor2Node::Ite => {
                let (c, _) = self.eval_bv(self.arg(line, 0)?)?;
                if c & 1 == 1 {
                    self.eval(self.arg(line, 1)?)
                } else {
                    self.eval(self.arg(line, 2)?)
                }
            }
            Btor2Node::Read => {
                let arr = self.eval(self.arg(line, 0)?)?;
                let (idx, _) = self.eval_bv(self.arg(line, 1)?)?;
                match arr {
                    OVal::Arr(a) => Ok(OVal::Bv {
                        bits: a.read(idx),
                        width: a.ew,
                    }),
                    OVal::Bv { .. } => Err("read on non-array".into()),
                }
            }
            Btor2Node::Write => {
                let arr = self.eval(self.arg(line, 0)?)?;
                let (idx, _) = self.eval_bv(self.arg(line, 1)?)?;
                let (val, _) = self.eval_bv(self.arg(line, 2)?)?;
                match arr {
                    OVal::Arr(a) => Ok(OVal::Arr(a.write(idx, val))),
                    OVal::Bv { .. } => Err("write on non-array".into()),
                }
            }

            other => Err(format!("op outside oracle whitelist: {other:?}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Preflight
// ---------------------------------------------------------------------------

struct Preflight {
    /// State node ids in declaration order.
    state_ids: Vec<NodeId>,
    /// state id -> init value node.
    init_of: HashMap<NodeId, NodeId>,
    /// state id -> next value node.
    next_of: HashMap<NodeId, NodeId>,
    /// Input node ids in declaration order, with widths.
    inputs: Vec<(NodeId, u32)>,
    /// No-`init` states with their nondet domain descriptor.
    nondet_slots: Vec<(NodeId, SlotKind)>,
    /// Total nondet frame-0 bits.
    frame0_bits: u64,
    /// Total input bits per step.
    input_bits: u64,
}

#[derive(Clone, Copy)]
enum SlotKind {
    Scalar(u32),
    Array { iw: u32, ew: u32 },
}

fn array_dims(sort: &Btor2Sort) -> Option<(u32, u32)> {
    let Btor2Sort::Array { index, element } = sort else {
        return None;
    };
    match (index.as_ref(), element.as_ref()) {
        (Btor2Sort::BitVec(iw), Btor2Sort::BitVec(ew)) => Some((*iw, *ew)),
        _ => None,
    }
}

fn preflight(program: &Btor2Program, config: &OracleConfig) -> Result<Preflight, String> {
    let mut state_ids = Vec::new();
    let mut init_of = HashMap::new();
    let mut next_of = HashMap::new();
    let mut inputs = Vec::new();
    let mut input_bits: u64 = 0;

    for line in &program.lines {
        match &line.node {
            Btor2Node::State(sort_id, _) => {
                match program.sorts.get(sort_id) {
                    Some(Btor2Sort::BitVec(_)) => {}
                    Some(sort @ Btor2Sort::Array { .. }) => {
                        if array_dims(sort).is_none() {
                            return Err("array state with non-bitvector index/element".into());
                        }
                    }
                    None => return Err(format!("state {} has undefined sort", line.id)),
                }
                state_ids.push(line.id);
            }
            Btor2Node::Input(sort_id, _) => match program.sorts.get(sort_id) {
                Some(Btor2Sort::BitVec(w)) => {
                    inputs.push((line.id, *w));
                    input_bits += u64::from(*w);
                }
                _ => {
                    return Err(
                        "array-sorted input (re-havocked every step — outside oracle class)".into(),
                    )
                }
            },
            Btor2Node::Init(_, state_id, value_id) => {
                init_of.insert(state_id.unsigned_abs() as i64, *value_id);
            }
            Btor2Node::Next(_, state_id, value_id) => {
                next_of.insert(state_id.unsigned_abs() as i64, *value_id);
            }
            _ => {}
        }
    }

    // DECLINE any state with no `next`: the lanes disagree on its semantics
    // (bitblast/word_replay HOLD, the CHC lane HAVOCS — see
    // docs/hwmcc/no-next-havoc-vs-hold-divergence.md). The oracle refuses to
    // pick a side rather than silently resolving the divergence.
    for &sid in &state_ids {
        if !next_of.contains_key(&sid) {
            return Err(format!(
                "state {sid} has no `next` line (havoc-vs-hold cross-lane divergence — declined)"
            ));
        }
    }

    if input_bits > config.max_input_bits_per_step {
        return Err(format!(
            "{input_bits} input bits per step exceeds cap {}",
            config.max_input_bits_per_step
        ));
    }

    // Nondet frame-0 slots.
    let sort_of_state = |sid: NodeId| -> Option<&Btor2Sort> {
        program.lines.iter().find_map(|l| match &l.node {
            Btor2Node::State(sort_id, _) if l.id == sid => program.sorts.get(sort_id),
            _ => None,
        })
    };
    let mut nondet_slots = Vec::new();
    let mut frame0_bits: u64 = 0;
    for &sid in &state_ids {
        if init_of.contains_key(&sid) {
            continue;
        }
        let sort = sort_of_state(sid).ok_or("state sort missing")?;
        match sort {
            Btor2Sort::BitVec(w) => {
                frame0_bits += u64::from(*w);
                nondet_slots.push((sid, SlotKind::Scalar(*w)));
            }
            s @ Btor2Sort::Array { .. } => {
                let (iw, ew) = array_dims(s).ok_or("bad array dims")?;
                let cells = 1u64
                    .checked_shl(iw)
                    .ok_or("array index width overflows enumeration")?;
                let bits = cells
                    .checked_mul(u64::from(ew))
                    .ok_or("array content bits overflow")?;
                frame0_bits += bits;
                nondet_slots.push((sid, SlotKind::Array { iw, ew }));
            }
        }
    }
    if frame0_bits > config.max_frame0_nondet_bits {
        return Err(format!(
            "{frame0_bits} nondeterministic frame-0 bits exceeds cap {}",
            config.max_frame0_nondet_bits
        ));
    }

    Ok(Preflight {
        state_ids,
        init_of,
        next_of,
        inputs,
        nondet_slots,
        frame0_bits,
        input_bits,
    })
}

// ---------------------------------------------------------------------------
// BFS exploration
// ---------------------------------------------------------------------------

enum Origin {
    /// A frame-0 state; carries the nondet (no-`init`) state values.
    Root(Vec<(NodeId, OVal)>),
    /// Produced from `parent` by one transition under `inputs`.
    Step {
        parent: usize,
        inputs: Vec<(NodeId, u128)>,
    },
}

fn node_cond(line_index: &HashMap<NodeId, &Btor2Line>, id: NodeId) -> Result<NodeId, String> {
    match line_index.get(&id).map(|l| &l.node) {
        Some(Btor2Node::Bad(c) | Btor2Node::Constraint(c)) => Ok(*c),
        _ => Err(format!("node {id} is not a bad/constraint")),
    }
}

fn oval_to_wordvalue(v: &OVal) -> WordValue {
    match v {
        OVal::Bv { bits, width } => WordValue::Bv {
            bits: *bits,
            width: *width,
        },
        OVal::Arr(a) => WordValue::Array {
            index_width: a.iw,
            elem_width: a.ew,
            default: a.default,
            cells: a.cells.iter().map(|(&k, &v)| (k, v)).collect(),
        },
    }
}

/// Run the K-step explicit-state oracle. See the module docs for semantics.
/// This is the differential harness's ground truth; it never feeds a
/// production verdict.
pub(crate) fn oracle_check(program: &Btor2Program, config: &OracleConfig) -> OracleOutcome {
    match oracle_check_inner(program, config) {
        Ok(outcome) => outcome,
        Err(reason) => OracleOutcome::Declined(reason),
    }
}

#[allow(clippy::too_many_lines)]
fn oracle_check_inner(
    program: &Btor2Program,
    config: &OracleConfig,
) -> Result<OracleOutcome, String> {
    let k = config.max_depth.min(64);
    let pf = preflight(program, config)?;
    let line_index: HashMap<NodeId, &Btor2Line> = program.lines.iter().map(|l| (l.id, l)).collect();

    if program.bad_properties.is_empty() {
        return Err("no bad properties (nothing to check)".into());
    }

    // Evaluate `init` values once, in an empty (const-only) context; a scalar
    // init of an array state broadcasts to a const-array.
    let empty_state: HashMap<NodeId, OVal> = HashMap::new();
    let empty_inputs: HashMap<NodeId, u128> = HashMap::new();
    let mut init_vals: HashMap<NodeId, OVal> = HashMap::new();
    for (&sid, &vid) in &pf.init_of {
        let mut ev = OracleEval::new(program, &line_index, &empty_state, &empty_inputs);
        let raw = ev
            .eval(vid)
            .map_err(|e| format!("non-constant init expression for state {sid}: {e}"))?;
        let sort = line_index
            .get(&sid)
            .and_then(|l| match &l.node {
                Btor2Node::State(sort_id, _) => program.sorts.get(sort_id),
                _ => None,
            })
            .ok_or("init of a non-state")?;
        let lifted = match (sort, raw) {
            (Btor2Sort::BitVec(w), OVal::Bv { bits, .. }) => OVal::Bv {
                bits: bits & mask(*w),
                width: *w,
            },
            (s @ Btor2Sort::Array { .. }, OVal::Bv { bits, .. }) => {
                let (iw, ew) = array_dims(s).ok_or("bad array dims")?;
                OVal::Arr(OArr {
                    iw,
                    ew,
                    default: bits & mask(ew),
                    cells: BTreeMap::new(),
                })
            }
            (Btor2Sort::Array { .. }, OVal::Arr(a)) => OVal::Arr(a),
            (Btor2Sort::BitVec(_), OVal::Arr(_)) => {
                return Err("array init value on a scalar state".into())
            }
        };
        init_vals.insert(sid, lifted);
    }

    // -- Frame-0 enumeration -------------------------------------------------
    let mut arena: Vec<(Vec<OVal>, Origin, usize)> = Vec::new(); // (state, origin, depth)
    let mut visited: HashSet<Vec<OVal>> = HashSet::new();
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut bytes_used: usize = 0;

    let decode_slot = |kind: SlotKind, bits: u64, cursor: &mut u32| -> OVal {
        match kind {
            SlotKind::Scalar(w) => {
                let v = (u128::from(bits) >> *cursor) & mask(w);
                *cursor += w;
                OVal::Bv { bits: v, width: w }
            }
            SlotKind::Array { iw, ew } => {
                let cells_n = 1u64 << iw;
                let mut cells = BTreeMap::new();
                for j in 0..cells_n {
                    let v = (u128::from(bits) >> *cursor) & mask(ew);
                    *cursor += ew;
                    if v != 0 {
                        cells.insert(u128::from(j), v);
                    }
                }
                let mut arr = OArr {
                    iw,
                    ew,
                    default: 0,
                    cells,
                };
                arr.canonicalize();
                OVal::Arr(arr)
            }
        }
    };

    let total_frame0 = pf.frame0_bits as u32;
    let frame0_count: u64 = 1u64 << total_frame0;
    for m in 0..frame0_count {
        let mut cursor = 0u32;
        let mut nondet: Vec<(NodeId, OVal)> = Vec::new();
        for &(sid, kind) in &pf.nondet_slots {
            nondet.push((sid, decode_slot(kind, m, &mut cursor)));
        }
        let nondet_map: HashMap<NodeId, &OVal> = nondet.iter().map(|(sid, v)| (*sid, v)).collect();
        let tuple: Vec<OVal> = pf
            .state_ids
            .iter()
            .map(|sid| {
                init_vals
                    .get(sid)
                    .cloned()
                    .or_else(|| nondet_map.get(sid).map(|v| (*v).clone()))
                    .ok_or_else(|| format!("state {sid} has neither init nor nondet slot"))
            })
            .collect::<Result<_, String>>()?;
        if visited.insert(tuple.clone()) {
            bytes_used += 2 * tuple.iter().map(OVal::approx_bytes).sum::<usize>() + 96;
            arena.push((tuple, Origin::Root(nondet), 0));
            queue.push_back(arena.len() - 1);
        }
    }

    // -- Input enumeration helper ---------------------------------------------
    let total_input_bits = pf.input_bits as u32;
    let input_count: u64 = 1u64 << total_input_bits;
    let decode_inputs = |m: u64| -> Vec<(NodeId, u128)> {
        let mut cursor = 0u32;
        pf.inputs
            .iter()
            .map(|&(iid, w)| {
                let v = (u128::from(m) >> cursor) & mask(w);
                cursor += w;
                (iid, v)
            })
            .collect()
    };

    // -- BFS -------------------------------------------------------------------
    let mut exhausted = true;
    while let Some(idx) = queue.pop_front() {
        let depth = arena[idx].2;
        let state_map: HashMap<NodeId, OVal> = pf
            .state_ids
            .iter()
            .cloned()
            .zip(arena[idx].0.iter().cloned())
            .collect();

        for m in 0..input_count {
            let inputs_vec = decode_inputs(m);
            let inputs_map: HashMap<NodeId, u128> = inputs_vec.iter().copied().collect();
            let mut ev = OracleEval::new(program, &line_index, &state_map, &inputs_map);

            // Constraints: assume semantics — a false constraint prunes this
            // (state, input) branch.
            let mut pruned = false;
            for &cid in &program.constraints {
                let cond = node_cond(&line_index, cid)?;
                let (c, _) = ev.eval(cond)?.as_bv()?;
                if c & 1 == 0 {
                    pruned = true;
                    break;
                }
            }
            if pruned {
                continue;
            }

            // Bad properties.
            let mut fired = Vec::new();
            for (j, &bid) in program.bad_properties.iter().enumerate() {
                let cond = node_cond(&line_index, bid)?;
                let (b, _) = ev.eval(cond)?.as_bv()?;
                if b & 1 == 1 {
                    fired.push(j);
                }
            }
            if !fired.is_empty() {
                let model = build_model(&arena, idx, depth, inputs_vec);
                cross_check_with_word_replay(program, &model, &fired);
                return Ok(OracleOutcome::Unsafe {
                    depth,
                    fired,
                    model,
                });
            }

            // Successor (simultaneous commit against the current frame).
            let next_tuple: Vec<OVal> = pf
                .state_ids
                .iter()
                .map(|sid| {
                    let vid = pf.next_of[sid];
                    ev.eval(vid)
                })
                .collect::<Result<_, String>>()?;
            if visited.contains(&next_tuple) {
                continue;
            }
            if depth >= k {
                // Depth bound hit with an unexplored successor: the bounded
                // verdict is exact but not a fixpoint.
                exhausted = false;
                continue;
            }
            if visited.len() >= MAX_VISITED_STATES {
                return Err("visited-set state-count cap exceeded".into());
            }
            let cost = 2 * next_tuple.iter().map(OVal::approx_bytes).sum::<usize>()
                + 96
                + inputs_vec.len() * 24;
            bytes_used += cost;
            if bytes_used > config.visited_byte_budget {
                return Err("visited-set byte budget exceeded".into());
            }
            visited.insert(next_tuple.clone());
            arena.push((
                next_tuple,
                Origin::Step {
                    parent: idx,
                    inputs: inputs_vec,
                },
                depth + 1,
            ));
            queue.push_back(arena.len() - 1);
        }
    }

    Ok(OracleOutcome::BoundedSafe {
        explored_depth: k,
        exhausted,
    })
}

/// Reconstruct the concrete trace (frame-0 nondet states + input frames) for a
/// bad firing at `idx` (depth `depth`) under final-frame inputs `final_inputs`.
fn build_model(
    arena: &[(Vec<OVal>, Origin, usize)],
    idx: usize,
    depth: usize,
    final_inputs: Vec<(NodeId, u128)>,
) -> WordLevelModel {
    let mut frames_rev: Vec<InputFrame> = Vec::with_capacity(depth + 1);
    frames_rev.push(final_inputs.into_iter().collect());
    let mut cur = idx;
    let initial = loop {
        match &arena[cur].1 {
            Origin::Step { parent, inputs } => {
                frames_rev.push(inputs.iter().copied().collect());
                cur = *parent;
            }
            Origin::Root(nondet) => {
                let mut init = InitialState::default();
                for (sid, v) in nondet {
                    init.states.insert(*sid, oval_to_wordvalue(v));
                }
                break init;
            }
        }
    };
    frames_rev.reverse();
    WordLevelModel {
        num_frames: frames_rev.len(),
        initial,
        input_frames: frames_rev,
    }
}

/// The free second opinion: every oracle-found trace must also replay to the
/// SAME bad properties through `word_replay`'s independent evaluator.
/// Build-failing on disagreement — that would mean one of the two concrete
/// interpreters (or the oracle's search bookkeeping) is wrong.
fn cross_check_with_word_replay(program: &Btor2Program, model: &WordLevelModel, fired: &[usize]) {
    let replay_fired =
        crate::word_replay::replay_collect_bad(program, &model.initial, &model.input_frames);
    assert_eq!(
        replay_fired.as_deref(),
        Some(fired),
        "ORACLE / word_replay DISAGREEMENT: the oracle's trace does not replay to the same \
         bad properties through word_replay — one of the two independent evaluators is wrong"
    );
}

// ---------------------------------------------------------------------------
// Tests (hand-computed tiny nets)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn check(net: &str) -> OracleOutcome {
        let prog = parse(net).expect("parse");
        oracle_check(&prog, &OracleConfig::default())
    }

    fn check_depth(net: &str, k: usize) -> OracleOutcome {
        let prog = parse(net).expect("parse");
        oracle_check(
            &prog,
            &OracleConfig {
                max_depth: k,
                ..OracleConfig::default()
            },
        )
    }

    /// Frozen array latch: mem init 0, next mem = mem, bad = (mem[i] == 1).
    /// SAFE at every depth; the frontier closes immediately.
    #[test]
    fn frozen_array_latch_safe() {
        let net = "\
1 sort bitvec 1
2 sort array 1 1
3 state 2 mem
4 zero 1
5 init 2 3 4
6 next 2 3 3
7 input 1 i
8 read 1 3 7
9 bad 8
";
        match check(net) {
            OracleOutcome::BoundedSafe {
                exhausted: true, ..
            } => {}
            other => panic!("expected exhausted BoundedSafe, got {other:?}"),
        }
    }

    /// Across-step write chain: a 2-bit counter walks the write index; the bad
    /// index (2) is only written at step 2 and only observable at frame 3.
    /// Hand-computed: UNSAFE at exactly depth 3. THE shape a single-step
    /// oracle cannot see (the write at step t is read at step t+1..).
    #[test]
    fn across_step_write_chain_unsafe_at_depth_3() {
        let net = "\
1 sort bitvec 2
2 sort bitvec 1
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
7 state 1 c
8 zero 1
9 init 1 7 8
10 one 1
11 add 1 7 10
12 next 1 7 11
13 one 2
14 write 3 4 7 13
15 next 3 4 14
16 constd 1 2
17 read 2 4 16
18 bad 17
";
        match check(net) {
            OracleOutcome::Unsafe { depth, fired, .. } => {
                assert_eq!(depth, 3, "cell 2 written at step 2, visible at frame 3");
                assert_eq!(fired, vec![0]);
            }
            other => panic!("expected Unsafe at depth 3, got {other:?}"),
        }
        // With K < 3, the same net is (exactly) bounded-safe.
        match check_depth(net, 2) {
            OracleOutcome::BoundedSafe {
                explored_depth: 2,
                exhausted: false,
            } => {}
            other => panic!("expected non-exhausted BoundedSafe at K=2, got {other:?}"),
        }
    }

    /// Aliasing read: write at input index i, read at input index j; bad
    /// requires (i == j held over the step via a latch) and observing the
    /// written value. Hand-computed UNSAFE at depth 1.
    #[test]
    fn aliasing_read_unsafe() {
        let net = "\
1 sort bitvec 1
2 sort array 1 1
3 state 2 mem
4 zero 1
5 init 2 3 4
6 input 1 i
7 state 1 iprev
8 init 1 7 4
9 next 1 7 6
10 one 1
11 write 2 3 6 10
12 next 2 3 11
13 read 1 3 7
14 bad 13
";
        // Step 0: write mem[i]=1, latch iprev=i. Frame 1: read mem[iprev] —
        // equals 1 iff iprev aliases the written index (always, here).
        match check(net) {
            OracleOutcome::Unsafe { depth, .. } => assert_eq!(depth, 1),
            other => panic!("expected Unsafe at depth 1, got {other:?}"),
        }
    }

    /// Extensionality pair: two arrays start equal (both const-0), one gets a
    /// write of 0 (a no-op write) — arrays stay extensionally equal, so
    /// bad = (a != b) is SAFE. Canonical maps make this exact.
    #[test]
    fn extensionality_noop_write_safe() {
        let net = "\
1 sort bitvec 1
2 sort array 1 1
3 state 2 a
4 state 2 b
5 zero 1
6 init 2 3 5
7 init 2 4 5
8 zero 1
9 write 2 3 8 8
10 next 2 3 9
11 next 2 4 4
12 neq 1 3 4
13 bad 12
";
        match check(net) {
            OracleOutcome::BoundedSafe {
                exhausted: true, ..
            } => {}
            other => panic!("expected exhausted BoundedSafe, got {other:?}"),
        }
    }

    /// Extensionality, unsafe twin: the write stores 1 instead of 0, so the
    /// arrays genuinely diverge at frame 1.
    #[test]
    fn extensionality_real_write_unsafe() {
        let net = "\
1 sort bitvec 1
2 sort array 1 1
3 state 2 a
4 state 2 b
5 zero 1
6 init 2 3 5
7 init 2 4 5
8 zero 1
14 one 1
9 write 2 3 8 14
10 next 2 3 9
11 next 2 4 4
12 neq 1 3 4
13 bad 12
";
        match check(net) {
            OracleOutcome::Unsafe { depth, .. } => assert_eq!(depth, 1),
            other => panic!("expected Unsafe at depth 1, got {other:?}"),
        }
    }

    /// Init-array vs nondeterministic-array: with NO init line the frame-0
    /// array contents are enumerated, so bad = (mem[0] == 1) fires already at
    /// depth 0 from a nonzero initial array; the init-0 twin is safe.
    #[test]
    fn nondet_array_vs_init_array() {
        let nondet = "\
1 sort bitvec 1
2 sort array 1 1
3 state 2 mem
6 next 2 3 3
4 zero 1
5 read 1 3 4
7 bad 5
";
        match check(nondet) {
            OracleOutcome::Unsafe { depth, model, .. } => {
                assert_eq!(depth, 0, "nondet initial contents fire immediately");
                // The model must carry the nonzero initial array (no-init
                // states are never defaulted).
                let arr = model.initial.states.values().next().expect("initial array");
                match arr {
                    WordValue::Array { default, cells, .. } => {
                        assert!(
                            *default != 0 || cells.values().any(|&v| v != 0),
                            "initial array must be nonzero to fire bad"
                        );
                    }
                    other => panic!("expected array initial state, got {other:?}"),
                }
            }
            other => panic!("expected Unsafe at depth 0, got {other:?}"),
        }

        let init0 = "\
1 sort bitvec 1
2 sort array 1 1
3 state 2 mem
8 zero 1
9 init 2 3 8
6 next 2 3 3
4 zero 1
5 read 1 3 4
7 bad 5
";
        match check(init0) {
            OracleOutcome::BoundedSafe {
                exhausted: true, ..
            } => {}
            other => panic!("expected exhausted BoundedSafe, got {other:?}"),
        }
    }

    /// Constraint pruning: bad = (x == 1) but a constraint forces x == 0.
    /// The branch is pruned, so the net is safe (assume semantics).
    #[test]
    fn constraint_prunes_bad_branch() {
        let net = "\
1 sort bitvec 1
2 input 1 x
3 state 1 s
4 zero 1
5 init 1 3 4
6 next 1 3 4
7 bad 2
8 not 1 2
9 constraint 8
";
        match check(net) {
            OracleOutcome::BoundedSafe { .. } => {}
            other => panic!("expected BoundedSafe, got {other:?}"),
        }
    }

    /// Decline: a state with no `next` line (the fenced havoc-vs-hold
    /// cross-lane divergence).
    #[test]
    fn declines_no_next_state() {
        let net = "\
1 sort bitvec 1
2 state 1 s
3 bad 2
";
        match check(net) {
            OracleOutcome::Declined(reason) => {
                assert!(reason.contains("no `next`"), "reason: {reason}");
            }
            other => panic!("expected Declined, got {other:?}"),
        }
    }

    /// Decline: array-sorted input (fresh array every step).
    #[test]
    fn declines_array_input() {
        let net = "\
1 sort bitvec 1
2 sort array 1 1
3 input 2 mem
4 zero 1
5 read 1 3 4
6 bad 5
";
        match check(net) {
            OracleOutcome::Declined(reason) => {
                assert!(reason.contains("array-sorted input"), "reason: {reason}");
            }
            other => panic!("expected Declined, got {other:?}"),
        }
    }

    /// Decline: op outside the whitelist (mul).
    #[test]
    fn declines_non_whitelisted_op() {
        let net = "\
1 sort bitvec 2
2 state 1 s
3 one 1
4 init 1 2 3
5 mul 1 2 2
6 next 1 2 5
7 constd 1 3
8 eq 1 2 7
9 sort bitvec 1
10 bad 8
";
        // NOTE: line 8's sort id (1) is 2 bits wide; eq produces 1 bit — the
        // oracle computes width from semantics (1), and declines only on the
        // mul. Keep the net well-formed for the parser regardless.
        match check(net) {
            OracleOutcome::Declined(reason) => {
                assert!(reason.contains("whitelist"), "reason: {reason}");
            }
            other => panic!("expected Declined, got {other:?}"),
        }
    }

    /// Decline: frame-0 nondet bits over the cap (a 4-bit-index array of 4-bit
    /// elements with no init = 64 nondet bits > 16).
    #[test]
    fn declines_over_cap_nondet() {
        let net = "\
1 sort bitvec 4
2 sort array 1 1
3 state 2 mem
4 next 2 3 3
5 zero 1
6 read 1 3 5
7 redor 1 6
8 sort bitvec 1
9 bad 7
";
        match check(net) {
            OracleOutcome::Declined(reason) => {
                assert!(reason.contains("frame-0 bits"), "reason: {reason}");
            }
            other => panic!("expected Declined, got {other:?}"),
        }
    }

    /// Canonicalization corner: on a tiny domain, an all-ones map re-roots to
    /// default=1 with no cells — the SAME canonical form however it was built.
    #[test]
    fn canonical_form_is_unique_on_tiny_domain() {
        let mut a = OArr {
            iw: 1,
            ew: 1,
            default: 0,
            cells: BTreeMap::from([(0u128, 1u128), (1u128, 1u128)]),
        };
        a.canonicalize();
        let b = OArr {
            iw: 1,
            ew: 1,
            default: 1,
            cells: BTreeMap::new(),
        };
        assert_eq!(a, b, "all-ones array must canonicalize to default=1");

        // And write-based construction reaches the same form.
        let c = OArr {
            iw: 1,
            ew: 1,
            default: 1,
            cells: BTreeMap::new(),
        }
        .write(0, 1); // no-op write
        assert_eq!(c, b);
    }
}
