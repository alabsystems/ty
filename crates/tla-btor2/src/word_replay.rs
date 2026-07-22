// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Word-level (bit-blast-free) concrete BTOR2 counterexample replay.
//!
//! This is the wide-index-array / bit-blast-*ineligible* sibling of
//! [`crate::witness`]. Where `witness.rs` replays a bit-level trace through the
//! bit-blasted AIGER circuit (and so must materialize each array as `1<<iw`
//! flat cells — impossible for a 16- or 32-bit index), this module replays a
//! **concrete word-level model** directly over the BTOR2 IR with:
//!
//! * bitvectors as width-masked [`u128`]s (all arithmetic wraps mod `2^width`), and
//! * arrays as sparse maps ([`WordValue::Array`]): a `default` plus explicit
//!   store overrides. A 32- or 64-bit index is just a `u128` map key, so a
//!   wide-index array replays with a handful of live cells and no `1<<iw` blowup.
//!
//! The concrete model comes from `ay-chc`'s derivation witness
//! ([`reconstruct_model`]): the solver's per-frame `SmtValue` array/scalar
//! assignments, re-keyed from CHC variable names to BTOR2 state/input node ids.
//!
//! ## Soundness (fail-closed)
//!
//! [`word_level_replay`] re-simulates `init`/`next`/`bad`/`constraint` forward
//! from the reconstructed frame-0 state and per-frame inputs (btorsim's exact
//! semantics) and reports success **only** if some `bad` literal is genuinely
//! true at some frame with every `constraint` satisfied up to that frame. A
//! model that is incomplete (a needed nondeterministic state or input is
//! missing, or an op it cannot evaluate) or that does not actually reach a bad
//! state yields `false` / `None` — never a witness that is not a real
//! counterexample. States with an `init` line take their frame-0 value from
//! `init` (deterministic); states without one are nondeterministic and must be
//! supplied by the reconstructed model, never defaulted to zero.

use std::collections::BTreeMap;

use ay_chc::SmtValue;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::to_chc::StateVarEntry;
use crate::types::{Btor2Line, Btor2Node, Btor2Program, Btor2Sort, NodeId};
use crate::witness::{binary_msb_first_from_int, Btor2Witness, StateValue};

// ---------------------------------------------------------------------------
// Value model
// ---------------------------------------------------------------------------

/// A concrete word-level value: a width-masked bitvector, or a functional array
/// (a `default` element plus sparse store overrides).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WordValue {
    /// Bitvector, `bits` masked to `width` bits.
    Bv {
        /// Value, masked to `width` bits.
        bits: u128,
        /// Bit width.
        width: u32,
    },
    /// Array: `read(a, j) = cells.get(j).unwrap_or(default)`; `write` clones and
    /// inserts. Index/element widths carried so cells render + mask correctly.
    Array {
        /// Index bit width.
        index_width: u32,
        /// Element bit width.
        elem_width: u32,
        /// Value of every index not present in `cells`.
        default: u128,
        /// Explicit store overrides (index -> element), last write wins.
        cells: FxHashMap<u128, u128>,
    },
}

impl WordValue {
    fn as_bv(&self) -> Option<(u128, u32)> {
        match self {
            WordValue::Bv { bits, width } => Some((*bits, *width)),
            WordValue::Array { .. } => None,
        }
    }
}

/// The nondeterministic frame-0 state supplied to the replay: one entry per
/// `state` line that has **no** `init` line (keyed by that state's BTOR2 node
/// id). States with an `init` line are computed from `init` and are absent here.
#[derive(Clone, Debug, Default)]
pub struct InitialState {
    /// State node id -> its frame-0 value.
    pub states: FxHashMap<NodeId, WordValue>,
}

/// One frame's input stimulus: input node id -> concrete value.
pub type InputFrame = FxHashMap<NodeId, u128>;

/// A reconstructed concrete counterexample model, ready for [`word_level_replay`]
/// / [`build_word_level_witness`]. Carried through
/// [`crate::Btor2CheckResult::Sat`] from the CHC lane; `None` on bit-level/GPU/
/// BMC lanes whose traces are already bit-level replayable.
#[derive(Clone, Debug)]
pub struct WordLevelModel {
    /// Frame count = max derivation level + 1 (transitions + 1).
    pub num_frames: usize,
    /// Frame-0 values for nondeterministic (no-`init`) states.
    pub initial: InitialState,
    /// Per-frame input stimulus, length `num_frames`.
    pub input_frames: Vec<InputFrame>,
}

// ---------------------------------------------------------------------------
// Bitvector helpers
// ---------------------------------------------------------------------------

fn mask(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

fn sign_bit_set(bits: u128, width: u32) -> bool {
    width > 0 && (bits >> (width - 1)) & 1 == 1
}

/// Two's-complement interpretation of `bits` at `width` as a signed 128-bit int.
fn to_signed(bits: u128, width: u32) -> i128 {
    let m = bits & mask(width);
    if sign_bit_set(m, width) && width < 128 {
        (m as i128) - (1i128 << width)
    } else {
        m as i128
    }
}

fn from_signed(v: i128, width: u32) -> u128 {
    (v as u128) & mask(width)
}

// ---------------------------------------------------------------------------
// Sort helpers
// ---------------------------------------------------------------------------

fn bv_width_of_sort(sort: &Btor2Sort) -> Option<u32> {
    match sort {
        Btor2Sort::BitVec(w) => Some(*w),
        Btor2Sort::Array { .. } => None,
    }
}

fn state_sort<'a>(program: &'a Btor2Program, state_node_id: NodeId) -> Option<&'a Btor2Sort> {
    for line in &program.lines {
        if line.id == state_node_id {
            if let Btor2Node::State(sort_id, _) = &line.node {
                return program.sorts.get(sort_id);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Concrete evaluator over the BTOR2 IR (one frame; memo cleared each frame)
// ---------------------------------------------------------------------------

struct Evaluator<'a> {
    program: &'a Btor2Program,
    line_index: &'a FxHashMap<NodeId, &'a Btor2Line>,
    state: &'a FxHashMap<NodeId, WordValue>,
    inputs: &'a FxHashMap<NodeId, u128>,
    /// Memo keyed by absolute node id (valid within one fixed frame).
    memo: FxHashMap<NodeId, WordValue>,
}

impl<'a> Evaluator<'a> {
    fn new(
        program: &'a Btor2Program,
        line_index: &'a FxHashMap<NodeId, &'a Btor2Line>,
        state: &'a FxHashMap<NodeId, WordValue>,
        inputs: &'a FxHashMap<NodeId, u128>,
    ) -> Self {
        Self {
            program,
            line_index,
            state,
            inputs,
            memo: FxHashMap::default(),
        }
    }

    fn sort(&self, sort_id: i64) -> Option<&Btor2Sort> {
        self.program.sorts.get(&sort_id)
    }

    fn result_width(&self, line: &Btor2Line) -> Option<u32> {
        bv_width_of_sort(self.sort(line.sort_id)?)
    }

    /// Evaluate an operand id, honoring the BTOR2 negative-id = bitwise-NOT
    /// shorthand (bitvectors only).
    fn eval(&mut self, node_id: NodeId) -> Option<WordValue> {
        let neg = node_id < 0;
        let abs = node_id.unsigned_abs() as i64;
        let base = self.eval_abs(abs)?;
        if neg {
            match base {
                WordValue::Bv { bits, width } => Some(WordValue::Bv {
                    bits: (!bits) & mask(width),
                    width,
                }),
                // Negating an array operand is not a valid BTOR2 construct.
                WordValue::Array { .. } => None,
            }
        } else {
            Some(base)
        }
    }

    fn eval_abs(&mut self, abs: NodeId) -> Option<WordValue> {
        if let Some(v) = self.memo.get(&abs) {
            return Some(v.clone());
        }
        let v = self.eval_abs_uncached(abs)?;
        self.memo.insert(abs, v.clone());
        Some(v)
    }

    fn arg(&mut self, line: &Btor2Line, idx: usize) -> Option<WordValue> {
        self.eval(*line.args.get(idx)?)
    }

    fn arg_bv(&mut self, line: &Btor2Line, idx: usize) -> Option<(u128, u32)> {
        self.arg(line, idx)?.as_bv()
    }

    #[allow(clippy::too_many_lines)]
    fn eval_abs_uncached(&mut self, abs: NodeId) -> Option<WordValue> {
        let line = *self.line_index.get(&abs)?;
        match &line.node {
            // State -> current frame value (populated for every state line).
            Btor2Node::State(_, _) => self.state.get(&abs).cloned(),

            // Input -> supplied stimulus (missing => cannot replay, fail-closed).
            Btor2Node::Input(sort_id, _) => {
                let w = bv_width_of_sort(self.sort(*sort_id)?)?;
                let v = self.inputs.get(&abs).copied()?;
                Some(WordValue::Bv {
                    bits: v & mask(w),
                    width: w,
                })
            }

            // -- Constants ---------------------------------------------------
            Btor2Node::Zero => Some(WordValue::Bv {
                bits: 0,
                width: self.result_width(line)?,
            }),
            Btor2Node::One => Some(WordValue::Bv {
                bits: 1 & mask(self.result_width(line)?),
                width: self.result_width(line)?,
            }),
            Btor2Node::Ones => {
                let w = self.result_width(line)?;
                Some(WordValue::Bv {
                    bits: mask(w),
                    width: w,
                })
            }
            Btor2Node::Const(bits) => {
                let w = self.result_width(line)?;
                let v = u128::from_str_radix(bits, 2).ok()?;
                Some(WordValue::Bv {
                    bits: v & mask(w),
                    width: w,
                })
            }
            Btor2Node::ConstD(dec) => {
                let w = self.result_width(line)?;
                let v = if let Some(stripped) = dec.strip_prefix('-') {
                    let abs_val: u128 = stripped.parse().ok()?;
                    from_signed(-(abs_val as i128), w)
                } else {
                    dec.parse::<u128>().ok()? & mask(w)
                };
                Some(WordValue::Bv { bits: v, width: w })
            }
            Btor2Node::ConstH(hex) => {
                let w = self.result_width(line)?;
                let v = u128::from_str_radix(hex, 16).ok()?;
                Some(WordValue::Bv {
                    bits: v & mask(w),
                    width: w,
                })
            }

            // -- Unary -------------------------------------------------------
            Btor2Node::Not => {
                let (a, w) = self.arg_bv(line, 0)?;
                Some(WordValue::Bv {
                    bits: (!a) & mask(w),
                    width: w,
                })
            }
            Btor2Node::Neg => {
                let (a, w) = self.arg_bv(line, 0)?;
                Some(WordValue::Bv {
                    bits: a.wrapping_neg() & mask(w),
                    width: w,
                })
            }
            Btor2Node::Inc => {
                let (a, w) = self.arg_bv(line, 0)?;
                Some(WordValue::Bv {
                    bits: a.wrapping_add(1) & mask(w),
                    width: w,
                })
            }
            Btor2Node::Dec => {
                let (a, w) = self.arg_bv(line, 0)?;
                Some(WordValue::Bv {
                    bits: a.wrapping_sub(1) & mask(w),
                    width: w,
                })
            }
            Btor2Node::Redand => {
                let (a, w) = self.arg_bv(line, 0)?;
                Some(bit(u128::from(a == mask(w))))
            }
            Btor2Node::Redor => {
                let (a, _) = self.arg_bv(line, 0)?;
                Some(bit(u128::from(a != 0)))
            }
            Btor2Node::Redxor => {
                let (a, _) = self.arg_bv(line, 0)?;
                Some(bit(u128::from(a.count_ones() & 1 == 1)))
            }

            // -- Binary arithmetic ------------------------------------------
            Btor2Node::Add => self.bin_arith(line, |a, b, w| a.wrapping_add(b) & mask(w)),
            Btor2Node::Sub => self.bin_arith(line, |a, b, w| a.wrapping_sub(b) & mask(w)),
            Btor2Node::Mul => self.bin_arith(line, |a, b, w| a.wrapping_mul(b) & mask(w)),
            Btor2Node::UDiv => self.bin_arith(line, |a, b, w| if b == 0 { mask(w) } else { a / b }),
            Btor2Node::URem => self.bin_arith(line, |a, b, _| if b == 0 { a } else { a % b }),
            Btor2Node::SDiv => self.bin_signed(line, smt_bvsdiv),
            Btor2Node::SRem => self.bin_signed(line, smt_bvsrem),
            Btor2Node::SMod => self.bin_signed(line, smt_bvsmod),

            // -- Binary bitwise ---------------------------------------------
            Btor2Node::And => self.bin_arith(line, |a, b, w| (a & b) & mask(w)),
            Btor2Node::Or => self.bin_arith(line, |a, b, w| (a | b) & mask(w)),
            Btor2Node::Xor => self.bin_arith(line, |a, b, w| (a ^ b) & mask(w)),
            Btor2Node::Nand => self.bin_arith(line, |a, b, w| (!(a & b)) & mask(w)),
            Btor2Node::Nor => self.bin_arith(line, |a, b, w| (!(a | b)) & mask(w)),
            Btor2Node::Xnor => self.bin_arith(line, |a, b, w| (!(a ^ b)) & mask(w)),

            // -- Shifts ------------------------------------------------------
            Btor2Node::Sll => self.bin_arith(line, |a, b, w| {
                if b >= u128::from(w) {
                    0
                } else {
                    (a << b) & mask(w)
                }
            }),
            Btor2Node::Srl => self.bin_arith(line, |a, b, w| {
                if b >= u128::from(w) {
                    0
                } else {
                    (a & mask(w)) >> b
                }
            }),
            Btor2Node::Sra => {
                let (a, w) = self.arg_bv(line, 0)?;
                let (b, _) = self.arg_bv(line, 1)?;
                let sh = if b >= u128::from(w) { w } else { b as u32 };
                let signed = to_signed(a, w);
                Some(WordValue::Bv {
                    bits: from_signed(signed >> sh.min(127), w),
                    width: w,
                })
            }
            Btor2Node::Rol => {
                let (a, w) = self.arg_bv(line, 0)?;
                let (b, _) = self.arg_bv(line, 1)?;
                Some(WordValue::Bv {
                    bits: rotl(a, b, w),
                    width: w,
                })
            }
            Btor2Node::Ror => {
                let (a, w) = self.arg_bv(line, 0)?;
                let (b, _) = self.arg_bv(line, 1)?;
                Some(WordValue::Bv {
                    bits: rotr(a, b, w),
                    width: w,
                })
            }

            // -- Comparisons (1-bit result) ---------------------------------
            Btor2Node::Eq => {
                let a = self.arg(line, 0)?;
                let b = self.arg(line, 1)?;
                Some(bit(u128::from(word_eq(&a, &b)?)))
            }
            Btor2Node::Neq => {
                let a = self.arg(line, 0)?;
                let b = self.arg(line, 1)?;
                Some(bit(u128::from(!word_eq(&a, &b)?)))
            }
            Btor2Node::Ult => self.cmp_u(line, |a, b| a < b),
            Btor2Node::Ulte => self.cmp_u(line, |a, b| a <= b),
            Btor2Node::Ugt => self.cmp_u(line, |a, b| a > b),
            Btor2Node::Ugte => self.cmp_u(line, |a, b| a >= b),
            Btor2Node::Slt => self.cmp_s(line, |a, b| a < b),
            Btor2Node::Slte => self.cmp_s(line, |a, b| a <= b),
            Btor2Node::Sgt => self.cmp_s(line, |a, b| a > b),
            Btor2Node::Sgte => self.cmp_s(line, |a, b| a >= b),

            // -- Concat / slice / extend ------------------------------------
            Btor2Node::Concat => {
                let (a, _) = self.arg_bv(line, 0)?;
                let (b, wb) = self.arg_bv(line, 1)?;
                let w = self.result_width(line)?;
                Some(WordValue::Bv {
                    bits: ((a << wb) | b) & mask(w),
                    width: w,
                })
            }
            Btor2Node::Slice(upper, lower) => {
                let (a, _) = self.arg_bv(line, 0)?;
                let w = upper - lower + 1;
                Some(WordValue::Bv {
                    bits: (a >> lower) & mask(w),
                    width: w,
                })
            }
            Btor2Node::Uext(n) => {
                let (a, aw) = self.arg_bv(line, 0)?;
                let w = aw + n;
                Some(WordValue::Bv {
                    bits: a & mask(w),
                    width: w,
                })
            }
            Btor2Node::Sext(n) => {
                let (a, aw) = self.arg_bv(line, 0)?;
                let w = aw + n;
                Some(WordValue::Bv {
                    bits: from_signed(to_signed(a, aw), w),
                    width: w,
                })
            }

            // -- Ternary / arrays -------------------------------------------
            Btor2Node::Ite => {
                let (c, _) = self.arg_bv(line, 0)?;
                if c & 1 == 1 {
                    self.arg(line, 1)
                } else {
                    self.arg(line, 2)
                }
            }
            Btor2Node::Read => {
                let arr = self.arg(line, 0)?;
                let (idx, _) = self.arg_bv(line, 1)?;
                match arr {
                    WordValue::Array {
                        index_width,
                        elem_width,
                        default,
                        cells,
                    } => {
                        let key = idx & mask(index_width);
                        let v = cells.get(&key).copied().unwrap_or(default);
                        Some(WordValue::Bv {
                            bits: v & mask(elem_width),
                            width: elem_width,
                        })
                    }
                    WordValue::Bv { .. } => None,
                }
            }
            Btor2Node::Write => {
                let arr = self.arg(line, 0)?;
                let (idx, _) = self.arg_bv(line, 1)?;
                let (val, _) = self.arg_bv(line, 2)?;
                match arr {
                    WordValue::Array {
                        index_width,
                        elem_width,
                        default,
                        mut cells,
                    } => {
                        cells.insert(idx & mask(index_width), val & mask(elem_width));
                        Some(WordValue::Array {
                            index_width,
                            elem_width,
                            default,
                            cells,
                        })
                    }
                    WordValue::Bv { .. } => None,
                }
            }

            // -- Boolean (1-bit) --------------------------------------------
            Btor2Node::Iff => {
                let (a, _) = self.arg_bv(line, 0)?;
                let (b, _) = self.arg_bv(line, 1)?;
                Some(bit(u128::from((a & 1) == (b & 1))))
            }
            Btor2Node::Implies => {
                let (a, _) = self.arg_bv(line, 0)?;
                let (b, _) = self.arg_bv(line, 1)?;
                Some(bit(u128::from((a & 1) == 0 || (b & 1) == 1)))
            }

            // -- Overflow predicates (1-bit) --------------------------------
            Btor2Node::Uaddo => {
                let (a, w) = self.arg_bv(line, 0)?;
                let (b, _) = self.arg_bv(line, 1)?;
                let (am, bm) = (a & mask(w), b & mask(w));
                // Carry out of `w` bits.
                let carry = am.checked_add(bm).map_or(true, |s| s > mask(w));
                Some(bit(u128::from(carry)))
            }
            Btor2Node::Usubo => {
                let (a, _) = self.arg_bv(line, 0)?;
                let (b, _) = self.arg_bv(line, 1)?;
                Some(bit(u128::from(a < b)))
            }
            Btor2Node::Umulo => {
                let (a, w) = self.arg_bv(line, 0)?;
                let (b, _) = self.arg_bv(line, 1)?;
                if w > 64 {
                    return None; // product may exceed u128 — fail-closed.
                }
                // Both operands fit in `w` <= 64 bits, so the product fits u128.
                let prod = (a & mask(w)) * (b & mask(w));
                Some(bit(u128::from(prod >> w != 0)))
            }
            Btor2Node::Saddo => {
                let (a, w) = self.arg_bv(line, 0)?;
                let (b, _) = self.arg_bv(line, 1)?;
                let s = to_signed(a, w) + to_signed(b, w);
                Some(bit(u128::from(s < min_signed(w) || s > max_signed(w))))
            }
            Btor2Node::Ssubo => {
                let (a, w) = self.arg_bv(line, 0)?;
                let (b, _) = self.arg_bv(line, 1)?;
                let s = to_signed(a, w) - to_signed(b, w);
                Some(bit(u128::from(s < min_signed(w) || s > max_signed(w))))
            }
            Btor2Node::Smulo => {
                let (a, w) = self.arg_bv(line, 0)?;
                let (b, _) = self.arg_bv(line, 1)?;
                if w > 63 {
                    return None; // signed product may exceed i128 — fail-closed.
                }
                let s = to_signed(a, w) * to_signed(b, w);
                Some(bit(u128::from(s < min_signed(w) || s > max_signed(w))))
            }
            Btor2Node::Sdivo => {
                let (a, w) = self.arg_bv(line, 0)?;
                let (b, _) = self.arg_bv(line, 1)?;
                Some(bit(u128::from(
                    to_signed(a, w) == min_signed(w) && to_signed(b, w) == -1,
                )))
            }

            // Not evaluable as a value.
            Btor2Node::SortBitVec(_)
            | Btor2Node::SortArray(_, _)
            | Btor2Node::Init(_, _, _)
            | Btor2Node::Next(_, _, _)
            | Btor2Node::Bad(_)
            | Btor2Node::Constraint(_)
            | Btor2Node::Fair(_)
            | Btor2Node::Justice(_)
            | Btor2Node::Output(_) => None,
        }
    }

    fn bin_arith(
        &mut self,
        line: &Btor2Line,
        f: impl Fn(u128, u128, u32) -> u128,
    ) -> Option<WordValue> {
        let (a, w) = self.arg_bv(line, 0)?;
        let (b, _) = self.arg_bv(line, 1)?;
        Some(WordValue::Bv {
            bits: f(a, b, w) & mask(w),
            width: w,
        })
    }

    fn bin_signed(
        &mut self,
        line: &Btor2Line,
        f: impl Fn(u128, u128, u32) -> u128,
    ) -> Option<WordValue> {
        let (a, w) = self.arg_bv(line, 0)?;
        let (b, _) = self.arg_bv(line, 1)?;
        Some(WordValue::Bv {
            bits: f(a, b, w) & mask(w),
            width: w,
        })
    }

    fn cmp_u(&mut self, line: &Btor2Line, f: impl Fn(u128, u128) -> bool) -> Option<WordValue> {
        let (a, _) = self.arg_bv(line, 0)?;
        let (b, _) = self.arg_bv(line, 1)?;
        Some(bit(u128::from(f(a, b))))
    }

    fn cmp_s(&mut self, line: &Btor2Line, f: impl Fn(i128, i128) -> bool) -> Option<WordValue> {
        let (a, w) = self.arg_bv(line, 0)?;
        let (b, _) = self.arg_bv(line, 1)?;
        Some(bit(u128::from(f(to_signed(a, w), to_signed(b, w)))))
    }
}

/// A 1-bit bitvector carrying `v & 1`.
fn bit(v: u128) -> WordValue {
    WordValue::Bv {
        bits: v & 1,
        width: 1,
    }
}

fn min_signed(w: u32) -> i128 {
    if w == 0 {
        0
    } else {
        -(1i128 << (w - 1))
    }
}

fn max_signed(w: u32) -> i128 {
    if w == 0 {
        0
    } else {
        (1i128 << (w - 1)) - 1
    }
}

fn rotl(a: u128, b: u128, w: u32) -> u128 {
    if w == 0 {
        return 0;
    }
    let s = (b % u128::from(w)) as u32;
    let a = a & mask(w);
    if s == 0 {
        a
    } else {
        ((a << s) | (a >> (w - s))) & mask(w)
    }
}

fn rotr(a: u128, b: u128, w: u32) -> u128 {
    if w == 0 {
        return 0;
    }
    let s = (b % u128::from(w)) as u32;
    let a = a & mask(w);
    if s == 0 {
        a
    } else {
        ((a >> s) | (a << (w - s))) & mask(w)
    }
}

/// SMT-LIB `bvsdiv`. Division by zero inherits `bvudiv`-by-zero (all ones).
fn smt_bvsdiv(a: u128, b: u128, w: u32) -> u128 {
    let sa = sign_bit_set(a, w);
    let sb = sign_bit_set(b, w);
    let na = if sa { a.wrapping_neg() & mask(w) } else { a };
    let nb = if sb { b.wrapping_neg() & mask(w) } else { b };
    let udiv = |x: u128, y: u128| if y == 0 { mask(w) } else { x / y };
    match (sa, sb) {
        (false, false) => udiv(na, nb),
        (true, false) => udiv(na, nb).wrapping_neg() & mask(w),
        (false, true) => udiv(na, nb).wrapping_neg() & mask(w),
        (true, true) => udiv(na, nb),
    }
}

/// SMT-LIB `bvsrem`: remainder with the sign of the dividend.
fn smt_bvsrem(a: u128, b: u128, w: u32) -> u128 {
    let sa = sign_bit_set(a, w);
    let sb = sign_bit_set(b, w);
    let na = if sa { a.wrapping_neg() & mask(w) } else { a };
    let nb = if sb { b.wrapping_neg() & mask(w) } else { b };
    let urem = if nb == 0 { na } else { na % nb };
    if sa {
        urem.wrapping_neg() & mask(w)
    } else {
        urem
    }
}

/// SMT-LIB `bvsmod`: modulus with the sign of the divisor.
fn smt_bvsmod(a: u128, b: u128, w: u32) -> u128 {
    let sa = to_signed(a, w);
    let sb = to_signed(b, w);
    if sb == 0 {
        return a & mask(w);
    }
    let r = sa.rem_euclid(sb.abs());
    let m = if sb < 0 && r != 0 { r - sb.abs() } else { r };
    from_signed(m, w)
}

/// Structural equality of two concrete word values (scalar or extensional array).
fn word_eq(a: &WordValue, b: &WordValue) -> Option<bool> {
    match (a, b) {
        (WordValue::Bv { bits: x, .. }, WordValue::Bv { bits: y, .. }) => Some(x == y),
        (
            WordValue::Array {
                index_width: iwa,
                elem_width: ewa,
                default: da,
                cells: ca,
            },
            WordValue::Array {
                default: db,
                cells: cb,
                ..
            },
        ) => {
            let em = mask(*ewa);
            let read = |cells: &FxHashMap<u128, u128>, default: u128, k: u128| {
                (cells.get(&k).copied().unwrap_or(default)) & em
            };
            if (da & em) != (db & em) {
                // Differing defaults: extensional equality decomposes over the
                // finite index domain `2^iw`. Let K = keys(ca) ∪ keys(cb)
                // (every key is already masked to `iw` at insertion — `Write`,
                // `smtvalue_to_wordvalue`, and the oracle's canonical form all
                // mask), and R = domain \ K. On R both sides read their
                // (differing masked) defaults, so:
                //
                //   a == b  ⟺  (∀k ∈ K: read_a(k) == read_b(k))  ∧  R = ∅
                //
                // R = ∅ ⟺ |K| == 2^iw — decided by arithmetic comparison
                // alone, never by enumerating the domain. This is EXACT for
                // arbitrary (non-canonical) cell maps: it needs no
                // most-frequent-value re-rooting, only the partition argument
                // above. For iw >= 128 the domain exceeds any in-memory map,
                // so R is necessarily nonempty and the sides differ there.
                if *iwa >= 128 {
                    return Some(false);
                }
                let domain: u128 = 1u128 << *iwa;
                // Dedup for the COUNT (chained iteration visits shared keys
                // twice — harmless for comparing, wrong for counting).
                let keys: FxHashSet<u128> = ca.keys().chain(cb.keys()).copied().collect();
                if (keys.len() as u128) < domain {
                    // Some residual index reads da != db: genuinely unequal
                    // (an exact answer, not a fail-closed approximation).
                    return Some(false);
                }
                // K covers the whole domain: the per-key loop below is total.
            }
            let keys = ca.keys().chain(cb.keys());
            for &k in keys {
                if read(ca, *da, k) != read(cb, *db, k) {
                    return Some(false);
                }
            }
            Some(true)
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// SmtValue -> WordValue reconstruction
// ---------------------------------------------------------------------------

fn smtvalue_to_u128(v: &SmtValue, width: u32) -> Option<u128> {
    match v {
        SmtValue::BitVec(val, _) => Some(val & mask(width)),
        SmtValue::Int(i) => Some((*i as u128) & mask(width)),
        SmtValue::Bool(b) => Some(u128::from(*b) & mask(width)),
        _ => None,
    }
}

fn smtvalue_to_wordvalue(v: &SmtValue, sort: &Btor2Sort) -> Option<WordValue> {
    match sort {
        Btor2Sort::BitVec(w) => Some(WordValue::Bv {
            bits: smtvalue_to_u128(v, *w)?,
            width: *w,
        }),
        Btor2Sort::Array { index, element } => {
            let iw = bv_width_of_sort(index)?;
            let ew = bv_width_of_sort(element)?;
            match v {
                SmtValue::ConstArray(default) => Some(WordValue::Array {
                    index_width: iw,
                    elem_width: ew,
                    default: smtvalue_to_u128(default, ew)?,
                    cells: FxHashMap::default(),
                }),
                SmtValue::ArrayMap { default, entries } => {
                    let mut cells = FxHashMap::default();
                    for (k, val) in entries {
                        // Later entries win (store-chain order).
                        cells.insert(smtvalue_to_u128(k, iw)?, smtvalue_to_u128(val, ew)?);
                    }
                    Some(WordValue::Array {
                        index_width: iw,
                        elem_width: ew,
                        default: smtvalue_to_u128(default, ew)?,
                        cells,
                    })
                }
                _ => None,
            }
        }
    }
}

/// Input node metadata: (CHC variable name, node id, bit width), in
/// declaration order, mirroring `to_chc`'s input naming.
fn input_meta(program: &Btor2Program) -> Vec<(String, NodeId, u32)> {
    let mut out = Vec::new();
    for line in &program.lines {
        if let Btor2Node::Input(sort_id, name) = &line.node {
            if let Some(Btor2Sort::BitVec(w)) = program.sorts.get(sort_id) {
                let nm = name.clone().unwrap_or_else(|| format!("i{}", line.id));
                out.push((nm, line.id, *w));
            }
        }
    }
    out
}

/// Reconstruct a concrete [`WordLevelModel`] from an `ay-chc` counterexample's
/// derivation witness. `state_vars` maps CHC variable names to state node ids;
/// `program` is the program that was translated (its `state`/`input` symbols
/// key the witness's raw `instances`). Returns `None` when there is no
/// derivation witness (nothing to reconstruct).
pub(crate) fn reconstruct_model(
    program: &Btor2Program,
    state_vars: &[StateVarEntry],
    cex: &ay_chc::Counterexample,
) -> Option<WordLevelModel> {
    let witness = cex.witness.as_ref()?;
    if witness.entries.is_empty() {
        return None;
    }

    // Merge instances per derivation level (first non-empty value per key wins).
    let mut by_level: BTreeMap<usize, FxHashMap<String, SmtValue>> = BTreeMap::new();
    for e in &witness.entries {
        let slot = by_level.entry(e.level).or_default();
        for (k, v) in &e.instances {
            slot.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }
    let max_level = *by_level.keys().max()?;
    let num_frames = max_level + 1;

    // Frame-0 (current-state) values, keyed by the state's raw CHC var name.
    let mut initial = InitialState::default();
    if let Some(level0) = by_level.get(&0) {
        for sv in state_vars {
            let Some(sort) = state_sort(program, sv.node_id) else {
                continue;
            };
            if let Some(val) = level0.get(&sv.var.name) {
                if let Some(wv) = smtvalue_to_wordvalue(val, sort) {
                    initial.states.insert(sv.node_id, wv);
                }
            }
        }
    }

    // Per-frame inputs: the transition producing state k+1 carries frame-k's
    // inputs, so they live in the level-(k+1) instance map.
    let inputs_meta = input_meta(program);
    let mut input_frames = Vec::with_capacity(num_frames);
    for k in 0..num_frames {
        let mut frame: InputFrame = FxHashMap::default();
        if let Some(inst) = by_level.get(&(k + 1)) {
            for (name, node_id, width) in &inputs_meta {
                if let Some(val) = inst.get(name) {
                    if let Some(u) = smtvalue_to_u128(val, *width) {
                        frame.insert(*node_id, u);
                    }
                }
            }
        }
        input_frames.push(frame);
    }

    Some(WordLevelModel {
        num_frames,
        initial,
        input_frames,
    })
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

/// Build the concrete frame-0 state for every `state` line: from its `init`
/// line when present (deterministic), otherwise from `initial_state`
/// (nondeterministic — never defaulted). Returns `None` if a nondeterministic
/// state has no supplied value or an `init` expression cannot be evaluated.
fn build_frame0_state(
    program: &Btor2Program,
    line_index: &FxHashMap<NodeId, &Btor2Line>,
    initial_state: &InitialState,
) -> Option<FxHashMap<NodeId, WordValue>> {
    // state_id -> init value node id.
    let mut init_of: FxHashMap<NodeId, NodeId> = FxHashMap::default();
    for line in &program.lines {
        if let Btor2Node::Init(_, state_id, value_id) = &line.node {
            init_of.insert(state_id.unsigned_abs() as i64, *value_id);
        }
    }

    let empty_state: FxHashMap<NodeId, WordValue> = FxHashMap::default();
    let empty_inputs: FxHashMap<NodeId, u128> = FxHashMap::default();

    let mut state: FxHashMap<NodeId, WordValue> = FxHashMap::default();
    for line in &program.lines {
        let Btor2Node::State(sort_id, _) = &line.node else {
            continue;
        };
        let sort = program.sorts.get(sort_id)?;
        let value = if let Some(&init_id) = init_of.get(&line.id) {
            // Init expressions are over constants; evaluate in an empty context.
            let mut ev = Evaluator::new(program, line_index, &empty_state, &empty_inputs);
            let raw = ev.eval(init_id)?;
            lift_init_value(sort, raw)?
        } else {
            initial_state.states.get(&line.id)?.clone()
        };
        state.insert(line.id, value);
    }
    Some(state)
}

/// Evaluate every `init` line's value in the empty context and return the
/// map `state_id -> lifted WordValue` for exactly the states that HAVE an
/// `init` (nondeterministic states are absent). `None` if any init expression
/// is not evaluable in the empty context (non-constant init — callers must
/// fail closed).
///
/// `pub(crate)`: the array-IC3 lane ([`crate::array_ic3`]) uses this for its
/// exact syntactic initiation check (its preflight guarantees const-support
/// init expressions, matching this evaluator's empty context).
pub(crate) fn eval_init_state_values(
    program: &Btor2Program,
) -> Option<FxHashMap<NodeId, WordValue>> {
    let line_index: FxHashMap<NodeId, &Btor2Line> =
        program.lines.iter().map(|l| (l.id, l)).collect();
    let mut init_of: FxHashMap<NodeId, NodeId> = FxHashMap::default();
    for line in &program.lines {
        if let Btor2Node::Init(_, state_id, value_id) = &line.node {
            init_of.insert(state_id.unsigned_abs() as i64, *value_id);
        }
    }
    let empty_state: FxHashMap<NodeId, WordValue> = FxHashMap::default();
    let empty_inputs: FxHashMap<NodeId, u128> = FxHashMap::default();
    let mut out: FxHashMap<NodeId, WordValue> = FxHashMap::default();
    for line in &program.lines {
        let Btor2Node::State(sort_id, _) = &line.node else {
            continue;
        };
        let Some(&init_id) = init_of.get(&line.id) else {
            continue;
        };
        let sort = program.sorts.get(sort_id)?;
        let mut ev = Evaluator::new(program, &line_index, &empty_state, &empty_inputs);
        let raw = ev.eval(init_id)?;
        out.insert(line.id, lift_init_value(sort, raw)?);
    }
    Some(out)
}

/// Lift a scalar `init` constant to a const-array when the state sort is an
/// array (BTOR2 `init <array> <state> <bv_const>`), mirroring `to_chc`.
fn lift_init_value(sort: &Btor2Sort, raw: WordValue) -> Option<WordValue> {
    match sort {
        Btor2Sort::Array { index, element } => {
            let iw = bv_width_of_sort(index)?;
            let ew = bv_width_of_sort(element)?;
            match raw {
                WordValue::Bv { bits, .. } => Some(WordValue::Array {
                    index_width: iw,
                    elem_width: ew,
                    default: bits & mask(ew),
                    cells: FxHashMap::default(),
                }),
                arr @ WordValue::Array { .. } => Some(arr),
            }
        }
        Btor2Sort::BitVec(_) => Some(raw),
    }
}

fn node_cond(program: &Btor2Program, line_id: NodeId) -> Option<NodeId> {
    for line in &program.lines {
        if line.id == line_id {
            return match &line.node {
                Btor2Node::Bad(c) | Btor2Node::Constraint(c) => Some(*c),
                _ => None,
            };
        }
    }
    None
}

/// Replay the concrete model forward and collect the bad-property indices that
/// fire (with every constraint satisfied up to that frame). `None` means the
/// model is incomplete or never reaches a bad state — fail-closed.
///
/// `pub(crate)`: the lazy-array BMC lane ([`crate::array_bmc`]) gates its SAT
/// claims on this replay (fired-property attribution), and the test-only
/// explicit-state oracle cross-checks its traces against it.
pub(crate) fn replay_collect_bad(
    program: &Btor2Program,
    initial_state: &InitialState,
    input_frames: &[InputFrame],
) -> Option<Vec<usize>> {
    let line_index: FxHashMap<NodeId, &Btor2Line> =
        program.lines.iter().map(|l| (l.id, l)).collect();

    // Precompute (state_id, next_value_id) for the transition relation.
    let next_of: Vec<(NodeId, NodeId)> = program
        .lines
        .iter()
        .filter_map(|line| match &line.node {
            Btor2Node::Next(_, state_id, value_id) => {
                Some((state_id.unsigned_abs() as i64, *value_id))
            }
            _ => None,
        })
        .collect();

    let mut state = build_frame0_state(program, &line_index, initial_state)?;

    for inputs in input_frames {
        let mut ev = Evaluator::new(program, &line_index, &state, inputs);

        // Every constraint must hold at this frame, else the trace is invalid.
        for &cline in &program.constraints {
            let cond = node_cond(program, cline)?;
            let (c, _) = ev.eval(cond)?.as_bv()?;
            if c & 1 == 0 {
                return None;
            }
        }

        // Any bad firing (with constraints held) is a genuine counterexample.
        let mut fired = Vec::new();
        for (j, &bline) in program.bad_properties.iter().enumerate() {
            let cond = node_cond(program, bline)?;
            let (b, _) = ev.eval(cond)?.as_bv()?;
            if b & 1 == 1 {
                fired.push(j);
            }
        }
        if !fired.is_empty() {
            return Some(fired);
        }

        // Simultaneous transition: evaluate all next RHS against the current
        // frame, then commit. States without a `next` line keep their value.
        let mut next_state = state.clone();
        for &(sid, value_id) in &next_of {
            let nv = ev.eval(value_id)?;
            next_state.insert(sid, nv);
        }
        drop(ev);
        state = next_state;
    }

    None
}

/// Fail-closed concrete replay: `true` iff replaying `initial_state` +
/// `input_frames` forward reaches a `bad` state with every `constraint`
/// satisfied at each frame, over concrete arrays-as-maps and
/// bitvectors-as-ints (wraparound arithmetic). No bit-blasting, so wide-index
/// arrays replay with a handful of live cells.
pub fn word_level_replay(
    program: &Btor2Program,
    initial_state: &InitialState,
    input_frames: &[InputFrame],
) -> bool {
    replay_collect_bad(program, initial_state, input_frames).is_some()
}

/// Render a concrete state value for the btorsim witness. Arrays emit one
/// `[index] value` line per explicit store cell (a const-array/no-write frame-0
/// value emits none — btorsim then uses the state's `init`); this is
/// index-width-agnostic (a wide-index array does not enumerate `1<<iw` cells).
fn render_state_value(v: &WordValue) -> Option<StateValue> {
    match v {
        WordValue::Bv { bits, width } => {
            Some(StateValue::BitVec(binary_msb_first_from_int(*bits, *width)))
        }
        WordValue::Array {
            index_width,
            elem_width,
            default,
            cells,
        } => {
            // The btorsim witness lists only explicit cells; every unlisted index
            // is then read by the replaying simulator as the state's init (0 here).
            // A NONZERO model default therefore cannot be faithfully serialized in
            // this format: `replay_collect_bad` used `default` for unlisted cells,
            // but a downstream simulator would read 0 — so the emitted witness
            // could FAIL to reach bad (a fake witness). Fail closed rather than
            // risk that. (ay-chc's BMC lane emits default 0, matching the implicit
            // 0, so this never fires in practice; it is a defensive soundness
            // guard — the alternative, enumerating 2^index_width cells, is exactly
            // the blowup this word-level lane exists to avoid.)
            if *default != 0 {
                return None;
            }
            let mut pairs: Vec<(u128, u128)> = cells.iter().map(|(&k, &v)| (k, v)).collect();
            pairs.sort_unstable();
            Some(StateValue::Array(
                pairs
                    .into_iter()
                    .map(|(k, val)| {
                        (
                            binary_msb_first_from_int(k, *index_width),
                            binary_msb_first_from_int(val, *elem_width),
                        )
                    })
                    .collect(),
            ))
        }
    }
}

/// Replay `model` over `program` and, only if a `bad` state is genuinely
/// reached (all constraints satisfied), project it to a standard
/// btorsim-compatible [`Btor2Witness`] (reusing the shared serializer). Returns
/// `None` — emit no witness — when the model is incomplete or does not replay to
/// a bad state (fail-closed: never a fabricated witness).
pub fn build_word_level_witness(
    program: &Btor2Program,
    model: &WordLevelModel,
) -> Option<Btor2Witness> {
    let fired = replay_collect_bad(program, &model.initial, &model.input_frames)?;
    if fired.is_empty() {
        return None;
    }

    let line_index: FxHashMap<NodeId, &Btor2Line> =
        program.lines.iter().map(|l| (l.id, l)).collect();
    let frame0 = build_frame0_state(program, &line_index, &model.initial)?;

    // Initial-state assignments, in state-declaration order.
    let mut init_state: Vec<(usize, String, StateValue)> = Vec::new();
    let mut ordinal = 0usize;
    for line in &program.lines {
        if let Btor2Node::State(_, name) = &line.node {
            let v = frame0.get(&line.id)?;
            let nm = name.clone().unwrap_or_else(|| format!("s{}", line.id));
            // Fail closed if any state value can't be faithfully serialized
            // (e.g. a nonzero-default array — see render_state_value): no witness
            // rather than a witness that might not replay to bad.
            init_state.push((ordinal, nm, render_state_value(v)?));
            ordinal += 1;
        }
    }

    // Per-frame input assignments.
    let inputs_meta = input_meta(program);
    let mut input_frames_render: Vec<Vec<(usize, String, String)>> = Vec::new();
    for frame in &model.input_frames {
        let mut fr = Vec::new();
        for (ord, (name, node_id, width)) in inputs_meta.iter().enumerate() {
            let val = frame.get(node_id).copied().unwrap_or(0);
            fr.push((ord, name.clone(), binary_msb_first_from_int(val, *width)));
        }
        input_frames_render.push(fr);
    }

    Some(Btor2Witness::from_parts(
        fired,
        init_state,
        input_frames_render,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    /// Wide-index (bv16) array net: `mem` init all-zero, `next` writes 5 into
    /// `mem[0]`, `bad = (mem[0] == 5)` fires one step later. Bit-blast-ineligible
    /// (index width 16 > 12), so this is exactly the word-level lane's class.
    const WIDE_ARRAY: &str = "\
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

    fn empty_model(num_frames: usize) -> WordLevelModel {
        WordLevelModel {
            num_frames,
            initial: InitialState::default(),
            input_frames: vec![FxHashMap::default(); num_frames],
        }
    }

    #[test]
    fn replays_wide_array_counterexample() {
        let prog = parse(WIDE_ARRAY).expect("parse");
        // 2 frames: frame 0 (mem all zero, bad false) -> transition writes 5 ->
        // frame 1 (mem[0]==5, bad fires).
        let model = empty_model(2);
        assert!(word_level_replay(
            &prog,
            &model.initial,
            &model.input_frames
        ));

        let w = build_word_level_witness(&prog, &model).expect("real cex -> witness");
        assert_eq!(w.bad_property_count(), 1);
        assert_eq!(w.frame_count(), 2);
        let text = w.to_btor2_string();
        assert!(text.starts_with("sat\nb0\n#0\n"), "header:\n{text}");
        assert!(text.trim_end().ends_with('.'), "terminator:\n{text}");
    }

    #[test]
    fn fail_closed_when_not_enough_frames() {
        let prog = parse(WIDE_ARRAY).expect("parse");
        // Only frame 0: mem all zero, bad = (mem[0]==5) is false, no transition
        // taken -> no bad reached -> no witness.
        let model = empty_model(1);
        assert!(!word_level_replay(
            &prog,
            &model.initial,
            &model.input_frames
        ));
        assert!(build_word_level_witness(&prog, &model).is_none());
    }

    #[test]
    fn safe_wide_array_never_replays_to_bad() {
        // Same shape but `bad = (mem[0] == 6)` while `next` writes 5 — SAFE, so
        // no reachable bad however many frames we replay.
        let safe = "\
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
        let prog = parse(safe).expect("parse");
        let model = empty_model(4);
        assert!(!word_level_replay(
            &prog,
            &model.initial,
            &model.input_frames
        ));
        assert!(build_word_level_witness(&prog, &model).is_none());
    }

    #[test]
    fn bv_wraparound_add() {
        // 8-bit counter: state c init 250, next = c + 10, bad = (c == 4).
        // 250 + 10 = 260 mod 256 = 4 -> bad fires at frame 1.
        let src = "\
1 sort bitvec 8
2 sort bitvec 1
3 state 1 c
4 constd 1 250
5 init 1 3 4
6 constd 1 10
7 add 1 3 6
8 next 1 3 7
9 constd 1 4
10 eq 2 3 9
11 bad 10
";
        let prog = parse(src).expect("parse");
        let model = empty_model(2);
        assert!(word_level_replay(
            &prog,
            &model.initial,
            &model.input_frames
        ));
        let w = build_word_level_witness(&prog, &model).expect("witness");
        let text = w.to_btor2_string();
        // Frame-0 scalar state emitted MSB-first: 250 = 11111010.
        assert!(text.contains("0 11111010 c\n"), "state line:\n{text}");
    }

    /// Defensive fail-closed guard (workflow adversarial-verify finding): a
    /// nonzero-default array model cannot be faithfully rendered in the btorsim
    /// per-cell witness format (a downstream simulator reads 0/init for unlisted
    /// cells, not the default), so `render_state_value` returns `None` and no
    /// witness is emitted — never a witness that could fail to replay to bad.
    #[test]
    fn nonzero_default_array_render_fails_closed() {
        let nonzero = WordValue::Array {
            index_width: 16,
            elem_width: 8,
            default: 7,
            cells: FxHashMap::default(),
        };
        assert!(
            render_state_value(&nonzero).is_none(),
            "nonzero-default array must NOT serialize (fail-closed)"
        );
        let zero = WordValue::Array {
            index_width: 16,
            elem_width: 8,
            default: 0,
            cells: FxHashMap::default(),
        };
        assert!(
            render_state_value(&zero).is_some(),
            "zero-default array must serialize"
        );
    }

    // -- word_eq extensional equality across differing defaults --------------
    //
    // Phase-1 pin (commit f297eb56, found by the oracle cross-check gate):
    // `word_eq` compared arrays with differing defaults as UNEQUAL even when
    // extensionally equal. Via the `Neq` replay arm that was a latent
    // wrong-UNSAFE hazard: for `bad = neq(a, b)` a spurious model whose arrays
    // are extensionally EQUAL (differing defaults, full-domain explicit-cell
    // cover) would REPLAY as bad-fired — the replay gate itself confirming a
    // false counterexample. The residual-domain rule makes word_eq exact:
    // a == b ⟺ all keys in K = keys(a) ∪ keys(b) agree AND (|K| == 2^iw OR
    // defaults agree).

    fn arr(iw: u32, ew: u32, default: u128, cells: &[(u128, u128)]) -> WordValue {
        WordValue::Array {
            index_width: iw,
            elem_width: ew,
            default,
            cells: cells.iter().copied().collect(),
        }
    }

    /// (1) The pinned case: iw=1, a = {default 0, cells {0:1, 1:1}} vs
    /// b = {default 1, no cells} — full-domain cover, extensionally equal.
    #[test]
    fn word_eq_differing_defaults_full_cover_is_equal() {
        let a = arr(1, 8, 0, &[(0, 1), (1, 1)]);
        let b = arr(1, 8, 1, &[]);
        assert_eq!(word_eq(&a, &b), Some(true), "extensionally equal (a vs b)");
        assert_eq!(word_eq(&b, &a), Some(true), "extensionally equal (b vs a)");
    }

    /// (2) Wide index, differing defaults, sparse cells: the residual domain
    /// is nonempty, so the arrays genuinely differ there.
    #[test]
    fn word_eq_differing_defaults_sparse_is_unequal() {
        let a = arr(16, 8, 0, &[(0, 1)]);
        let b = arr(16, 8, 1, &[(0, 1)]);
        assert_eq!(word_eq(&a, &b), Some(false));
        // Degenerate width guard: iw >= 128 always has residual domain.
        let wa = arr(128, 8, 0, &[]);
        let wb = arr(128, 8, 1, &[]);
        assert_eq!(word_eq(&wa, &wb), Some(false));
    }

    /// (3) Same defaults but an explicit-cell disagreement stays unequal.
    #[test]
    fn word_eq_same_default_cell_disagreement_is_unequal() {
        let a = arr(16, 8, 0, &[(3, 5)]);
        let b = arr(16, 8, 0, &[(3, 6)]);
        assert_eq!(word_eq(&a, &b), Some(false));
        // And full-domain cover with a cell mismatch (differing defaults).
        let c = arr(1, 8, 0, &[(0, 1), (1, 2)]);
        let d = arr(1, 8, 1, &[(0, 1)]);
        assert_eq!(word_eq(&c, &d), Some(false));
    }

    /// Edge cases: iw=0 (domain size 1) and element-width masking.
    #[test]
    fn word_eq_edge_domains_and_masking() {
        // iw=0: single-index domain; one explicit cell covers it entirely.
        let a = arr(0, 8, 3, &[(0, 7)]);
        let b = arr(0, 8, 7, &[]);
        assert_eq!(word_eq(&a, &b), Some(true));
        // Defaults equal only after ew masking (0x13 & 0xF == 0x03): the
        // equal-defaults path applies, no full-cover requirement.
        let c = arr(16, 4, 0x13, &[]);
        let d = arr(16, 4, 0x03, &[]);
        assert_eq!(word_eq(&c, &d), Some(true));
    }

    /// (4) The severity-upgrade regression: `bad = neq(a, b)` over two
    /// nondeterministic array states. With extensionally-EQUAL arrays that
    /// differ in default (full-domain cover), replay must NOT confirm bad —
    /// pre-fix it did (false counterexample through the replay gate itself).
    /// The genuinely-unequal twin must still fire.
    #[test]
    fn neq_of_extensionally_equal_arrays_does_not_fire() {
        let src = "\
1 sort bitvec 1
2 sort bitvec 8
3 sort array 1 2
4 state 3 a
5 state 3 b
6 neq 1 4 5
7 bad 6
";
        let prog = parse(src).expect("parse");

        // Extensionally equal: a = {d:0, cells {0:1, 1:1}}, b = {d:1, {}}.
        let mut equal = InitialState::default();
        equal.states.insert(4, arr(1, 8, 0, &[(0, 1), (1, 1)]));
        equal.states.insert(5, arr(1, 8, 1, &[]));
        assert!(
            !word_level_replay(&prog, &equal, &[FxHashMap::default()]),
            "neq of extensionally equal arrays must not replay to bad"
        );

        // Genuinely unequal twin (cell 1 differs): bad must fire at frame 0.
        let mut unequal = InitialState::default();
        unequal.states.insert(4, arr(1, 8, 0, &[(0, 1), (1, 2)]));
        unequal.states.insert(5, arr(1, 8, 1, &[]));
        assert_eq!(
            replay_collect_bad(&prog, &unequal, &[FxHashMap::default()]),
            Some(vec![0]),
            "neq of genuinely unequal arrays must fire"
        );

        // And the eq twin: extensionally equal arrays DO satisfy eq-shaped bad.
        let eq_src = "\
1 sort bitvec 1
2 sort bitvec 8
3 sort array 1 2
4 state 3 a
5 state 3 b
6 eq 1 4 5
7 bad 6
";
        let eq_prog = parse(eq_src).expect("parse");
        assert_eq!(
            replay_collect_bad(&eq_prog, &equal, &[FxHashMap::default()]),
            Some(vec![0]),
            "eq of extensionally equal arrays must fire post-fix"
        );
    }
}
