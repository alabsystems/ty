// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! BTOR2 to AIGER bit-blasting translator.
//!
//! Converts a word-level BTOR2 program into a bit-level AIGER circuit by
//! expanding each N-bit bitvector variable into N boolean (1-bit) AIG signals.
//! BTOR2 operations become networks of AND-inverter gates.
//!
//! This enables the fast SAT-based IC3/PDR engine (tla-aiger) to solve narrow
//! bitvector benchmarks that are slow via the CHC path.
//!
//! Supported operations:
//! - Boolean/bitwise: not, and, or, xor, nand, nor, xnor
//! - Comparison: eq, neq, ult, ulte, ugt, ugte, slt, slte, sgt, sgte
//! - Arithmetic: add, sub, neg, inc, dec
//! - Indexing: slice, uext, sext, concat
//! - Control: ite
//! - Reduction: redand, redor, redxor
//! - Shifts: sll, srl, sra with constant shift amounts
//! - Constants: zero, one, ones, const, constd, consth
//!
//! Not yet supported (returns error):
//! - Multiplication, division, remainder (mul, udiv, sdiv, urem, srem, smod)
//! - Array operations (read, write)
//! - Overflow detection (saddo, sdivo, smulo, ssubo, uaddo, umulo, usubo)
//! - Rotate (rol, ror)
//! - Dynamic shifts where the shift amount is not statically constant
//! - Non-constant `init` values (gate expressions): the AIGER latch reset is
//!   per-bit constant-or-nondeterministic, which cannot express cross-bit
//!   correlations of an expression init; such nets are rejected (fail-closed)
//!   and routed to the exact word-level CHC lane

use std::collections::HashMap;

use crate::error::{Btor2Error, MAX_BV_WIDTH};
use crate::types::{Btor2Node, Btor2Program, Btor2Sort, NodeId};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Result of bit-blasting: an AIGER-compatible circuit representation.
///
/// Uses the same literal encoding as AIGER: variable = lit/2, negated = lit%2.
/// Literal 0 = constant FALSE, literal 1 = constant TRUE.
#[derive(Debug, Clone)]
pub struct BitblastedCircuit {
    /// Maximum variable index (all vars in 1..=max_var).
    pub max_var: u64,
    /// Input literals (one per bit of each BTOR2 input). Always even.
    pub inputs: Vec<u64>,
    /// Latch definitions: (current_lit, next_lit, reset_value).
    /// current_lit is always even (positive literal for the latch variable).
    /// reset_value: 0 = reset to 0, 1 = reset to 1, current_lit = nondeterministic.
    pub latches: Vec<(u64, u64, u64)>,
    /// AND gates: (lhs, rhs0, rhs1) where lhs = rhs0 AND rhs1.
    /// lhs is always even.
    pub ands: Vec<(u64, u64, u64)>,
    /// Bad-state property literals (one per BTOR2 `bad` property).
    pub bad: Vec<u64>,
    /// Constraint literals (one per BTOR2 `constraint`).
    pub constraints: Vec<u64>,
    /// Per BTOR2 STATE variable, in program order: `(name, bit literals
    /// LSB-first)`. Names follow the CHC translation's convention (the node
    /// symbol, or `s{k}` by state-declaration order), so a bit-level trace
    /// mapped through this table produces the same word-level assignment
    /// keys the CHC counterexamples use. Array states carry their flat
    /// expanded bit vector.
    pub state_bits: Vec<(String, Vec<u64>)>,
    /// Per BTOR2 INPUT, in program order: `(name, bit literals LSB-first)`,
    /// named by symbol or `i{k}`.
    pub input_bits: Vec<(String, Vec<u64>)>,
}

/// Bit-blast a BTOR2 program into an AIGER-compatible circuit.
///
/// `max_width` caps the per-variable bit width; it is itself clamped to
/// [`MAX_BV_WIDTH`] so the `u128`-backed constant encoding can never overflow.
///
/// # Errors
///
/// Returns [`Btor2Error`] if the program uses an operation not supported by the
/// bit-blaster (arrays, multiplication/division/remainder, overflow detection,
/// rotates, dynamic-amount shifts — see the module docs) or contains a bitvector
/// wider than the effective width cap.
pub fn bitblast(program: &Btor2Program, max_width: u32) -> Result<BitblastedCircuit, Btor2Error> {
    // Re-assert the hard MAX_BV_WIDTH cap independent of the caller-supplied
    // `max_width`: every constant is materialized through a u128 (see
    // `bits_from_u128` / `const_from_decimal` / `const_from_hex`), so a width
    // above 128 would truncate/mask mod 2^128 and yield a wrong-model verdict.
    // Clamp `max_width` so it can never exceed MAX_BV_WIDTH.
    let effective_max = max_width.min(MAX_BV_WIDTH);
    let mut ctx = BitblastContext::new(effective_max);
    ctx.translate(program)?;
    Ok(ctx.finish())
}

/// Check whether a BTOR2 program is eligible for bit-blasting, without building
/// the circuit.
///
/// Returns `Ok(max_bv_width)` — the widest bitvector encountered — if every
/// operation is supported and no width exceeds the effective cap (the smaller of
/// `max_width` and [`MAX_BV_WIDTH`]).
///
/// # Errors
///
/// Returns `Err(String)` naming the first unsupported operation or
/// over-wide/array sort that makes the program ineligible.
pub fn bitblast_eligible(program: &Btor2Program, max_width: u32) -> Result<u32, String> {
    let mut max_bv = 0u32;
    // Re-assert the hard MAX_BV_WIDTH cap independent of the caller-supplied
    // `max_width`: a width above 128 cannot be modeled by the u128-backed const
    // materialization, so it is ineligible regardless of the requested bound.
    let effective_max = max_width.min(MAX_BV_WIDTH);

    // Node ids whose defining line is a CONSTANT (const / constd / consth /
    // zero / one / ones). Every `init` value must resolve to one of these
    // (possibly negated): the AIGER latch model can only express a reset of
    // constant-0, constant-1, or fully nondeterministic per bit. A gate-
    // expression init (e.g. `s = concat(i, i)`, which forces bit0 == bit1)
    // has cross-bit correlations no per-bit constant reset can express, and
    // approximating it with a nondeterministic reset OVER-approximates the
    // initial-state set — which can manufacture a spurious counterexample
    // (`sat`) on a genuinely safe net. Fail closed here so the auto-router
    // falls through to the word-level CHC lane, which encodes init values
    // exactly as expressions.
    let constant_ids: std::collections::HashSet<NodeId> = program
        .lines
        .iter()
        .filter(|line| {
            matches!(
                &line.node,
                Btor2Node::Const(_)
                    | Btor2Node::ConstD(_)
                    | Btor2Node::ConstH(_)
                    | Btor2Node::Zero
                    | Btor2Node::One
                    | Btor2Node::Ones
            )
        })
        .map(|line| line.id)
        .collect();

    for line in &program.lines {
        // Reject non-constant init values (see `constant_ids` above). A
        // negated reference (negative id) to a constant is still a constant.
        if let Btor2Node::Init(_, _, value_ref) = &line.node {
            if !constant_ids.contains(&value_ref.abs()) {
                return Err(format!(
                    "init value (node {}) is a gate expression, not a constant; \
                     bit-blasting would over-approximate the initial-state set — \
                     routing to the word-level CHC lane",
                    value_ref.abs()
                ));
            }
        }

        // Check sort widths
        if let Some(sort) = program.sorts.get(&line.sort_id) {
            match sort {
                Btor2Sort::BitVec(w) => {
                    if *w > effective_max {
                        return Err(format!(
                            "bitvector width {w} exceeds max_width {effective_max}"
                        ));
                    }
                    max_bv = max_bv.max(*w);
                }
                sort @ Btor2Sort::Array { .. } => {
                    let (iw, ew, _num, flat) = array_dims_of(sort).ok_or_else(|| {
                        "array with non-bitvector index/element not supported for bit-blasting"
                            .to_string()
                    })?;
                    if iw > ARRAY_INDEX_MAX_BITS {
                        return Err(format!(
                            "array index width {iw} exceeds expansion limit {ARRAY_INDEX_MAX_BITS}"
                        ));
                    }
                    if ew > effective_max {
                        return Err(format!(
                            "array element width {ew} exceeds max_width {effective_max}"
                        ));
                    }
                    if flat > ARRAY_FLAT_MAX_BITS {
                        return Err(format!(
                            "expanded array width {flat} exceeds limit {ARRAY_FLAT_MAX_BITS}"
                        ));
                    }
                    max_bv = max_bv.max(ew);
                }
            }
        }

        // Every SCALAR operator now bit-blasts — including rotates (constant OR
        // variable amount, via a barrel rotator) and shifts (barrel shifter).
        // Only the array/width limits above can still decline a node.
    }

    Ok(max_bv)
}

fn bits_to_shift_amount(bits: &[u64]) -> Result<usize, String> {
    let mut amount = 0usize;
    for (idx, bit) in bits.iter().copied().enumerate() {
        match bit {
            0 => {}
            1 => {
                if idx >= usize::BITS as usize {
                    return Err("constant shift amount exceeds host usize width".into());
                }
                amount = amount.saturating_add(1usize << idx);
            }
            lit => {
                return Err(format!(
                    "shift amount contains non-constant literal {lit} at bit {idx}"
                ));
            }
        }
    }
    Ok(amount)
}

// ---------------------------------------------------------------------------
// Array expansion (bounded theory-of-arrays → bit-level)
// ---------------------------------------------------------------------------
//
// Small arrays are bit-blasted by EXPANSION: an array of 2^k elements each
// `ew` bits wide becomes a flat 2^k*ew-bit signal (element `e` occupies bits
// [e*ew, e*ew+ew)). A `read` is a one-hot mux selected by the index; a `write`
// conditionally updates each element. Once expanded, every other op (state,
// next, init, ite, eq) treats the array as an ordinary wide bitvector, so no
// other handler needs to know about arrays. This is the same array-blasting
// rIC3 uses; it is sound and complete for arrays small enough to expand.

/// Max array index width we expand (2^width elements).
const ARRAY_INDEX_MAX_BITS: u32 = 12;
/// Max total expanded array width in bits — guards combinational blow-up.
const ARRAY_FLAT_MAX_BITS: u32 = 8192;

/// For an array sort with bitvector index and element, return
/// `(index_width, element_width, num_elements, flat_width)`. `None` if the
/// sort is not an array, or its index/element are not plain bitvectors, or the
/// expansion would overflow.
fn array_dims_of(sort: &Btor2Sort) -> Option<(u32, u32, usize, u32)> {
    let Btor2Sort::Array { index, element } = sort else {
        return None;
    };
    let Btor2Sort::BitVec(iw) = index.as_ref() else {
        return None;
    };
    let Btor2Sort::BitVec(ew) = element.as_ref() else {
        return None;
    };
    let num = 1usize.checked_shl(*iw)?;
    let flat = (num as u64).checked_mul(u64::from(*ew))?;
    let flat = u32::try_from(flat).ok()?;
    Some((*iw, *ew, num, flat))
}

/// Flat bit-width of an expanded array sort.
fn array_flat_width(sort: &Btor2Sort) -> Option<u32> {
    array_dims_of(sort).map(|(_, _, _, flat)| flat)
}

// ---------------------------------------------------------------------------
// Internal context
// ---------------------------------------------------------------------------

/// A bitvector signal: a vector of AIGER literals (LSB first).
type BvSignal = Vec<u64>;

struct BitblastContext {
    /// Next available variable index.
    next_var: u64,
    /// Maximum bitvector width allowed.
    max_width: u32,
    /// Maps BTOR2 node ID to its bit-blasted signal (vector of AIGER lits).
    signals: HashMap<NodeId, BvSignal>,
    /// AIGER inputs (flat list of input literals).
    inputs: Vec<u64>,
    /// Latch definitions.
    latches: Vec<(u64, u64, u64)>,
    /// AND gate definitions.
    ands: Vec<(u64, u64, u64)>,
    /// Bad-state property literals.
    bad: Vec<u64>,
    /// Constraint literals.
    constraints: Vec<u64>,
    /// Maps BTOR2 state node ID to latch bit literals (current state).
    state_lits: HashMap<NodeId, BvSignal>,
    /// `(name, bit literals)` per state, program order (witness mapping).
    state_bits: Vec<(String, BvSignal)>,
    /// `(name, bit literals)` per input, program order (witness mapping).
    input_bits: Vec<(String, BvSignal)>,
}

impl BitblastContext {
    fn new(max_width: u32) -> Self {
        BitblastContext {
            next_var: 1, // Var 0 is reserved for constant FALSE
            max_width,
            signals: HashMap::new(),
            inputs: Vec::new(),
            latches: Vec::new(),
            ands: Vec::new(),
            bad: Vec::new(),
            constraints: Vec::new(),
            state_lits: HashMap::new(),
            state_bits: Vec::new(),
            input_bits: Vec::new(),
        }
    }

    /// Allocate a fresh variable and return its positive literal.
    fn alloc_var(&mut self) -> u64 {
        let lit = self.next_var << 1;
        self.next_var += 1;
        lit
    }

    /// Negate a literal.
    #[inline]
    fn neg(lit: u64) -> u64 {
        lit ^ 1
    }

    /// Create an AND gate: returns lhs literal where lhs = a AND b.
    fn mk_and(&mut self, a: u64, b: u64) -> u64 {
        // Constant propagation
        if a == 0 || b == 0 {
            return 0; // FALSE
        }
        if a == 1 {
            return b;
        }
        if b == 1 {
            return a;
        }
        if a == b {
            return a;
        }
        if a == Self::neg(b) {
            return 0; // a AND !a = FALSE
        }

        let lhs = self.alloc_var();
        self.ands.push((lhs, a, b));
        lhs
    }

    /// Create an OR gate: a OR b = NOT(NOT(a) AND NOT(b)).
    fn mk_or(&mut self, a: u64, b: u64) -> u64 {
        // Constant propagation
        if a == 1 || b == 1 {
            return 1; // TRUE
        }
        if a == 0 {
            return b;
        }
        if b == 0 {
            return a;
        }
        if a == b {
            return a;
        }
        if a == Self::neg(b) {
            return 1; // a OR !a = TRUE
        }

        Self::neg(self.mk_and(Self::neg(a), Self::neg(b)))
    }

    /// Create an XOR gate: a XOR b = (a AND !b) OR (!a AND b).
    fn mk_xor(&mut self, a: u64, b: u64) -> u64 {
        if a == b {
            return 0;
        }
        if a == Self::neg(b) {
            return 1;
        }
        if a == 0 {
            return b;
        }
        if a == 1 {
            return Self::neg(b);
        }
        if b == 0 {
            return a;
        }
        if b == 1 {
            return Self::neg(a);
        }

        let and1 = self.mk_and(a, Self::neg(b));
        let and2 = self.mk_and(Self::neg(a), b);
        self.mk_or(and1, and2)
    }

    /// Create a MUX: if sel then a else b.
    fn mk_mux(&mut self, sel: u64, a: u64, b: u64) -> u64 {
        if sel == 1 {
            return a;
        }
        if sel == 0 {
            return b;
        }
        if a == b {
            return a;
        }

        // mux = (sel AND a) OR (!sel AND b)
        let t = self.mk_and(sel, a);
        let f = self.mk_and(Self::neg(sel), b);
        self.mk_or(t, f)
    }

    /// 1-bit signal that is TRUE iff `index` equals the constant `val`
    /// (AND of per-bit matches; constant-folded by `mk_and`).
    fn index_eq_const(&mut self, index: &BvSignal, val: usize) -> u64 {
        let mut acc = 1u64; // TRUE
        for (i, &bit) in index.iter().enumerate() {
            let want = if i < usize::BITS as usize {
                (val >> i) & 1
            } else {
                0
            };
            let lit = if want == 1 { bit } else { Self::neg(bit) };
            acc = self.mk_and(acc, lit);
        }
        acc
    }

    /// If every bit of `index` is a constant literal (FALSE=0 or TRUE=1),
    /// return its concrete value; otherwise `None` (a symbolic bit is present).
    /// Lets `array_read`/`array_write` wire a constant address directly instead
    /// of emitting an `n`-way one-hot structure that would only fold away later.
    fn const_index_value(index: &BvSignal) -> Option<usize> {
        let mut val = 0usize;
        for (i, &bit) in index.iter().enumerate() {
            match bit {
                0 => {} // FALSE
                1 => {
                    if i >= usize::BITS as usize {
                        return None; // would overflow — let the mux handle it
                    }
                    val |= 1usize << i;
                }
                _ => return None, // symbolic bit
            }
        }
        Some(val)
    }

    /// Array select: one-hot mux of element `index` out of `n` flat elements,
    /// each `ew` bits wide (element `e` at flat bits `[e*ew, e*ew+ew)`).
    fn array_read(&mut self, array: &BvSignal, index: &BvSignal, ew: usize, n: usize) -> BvSignal {
        // Constant address: return the element wires directly (an out-of-range
        // index reads 0, matching the one-hot mux with no matching selector).
        if let Some(e) = Self::const_index_value(index) {
            return if e < n {
                array[e * ew..(e + 1) * ew].to_vec()
            } else {
                vec![0u64; ew]
            };
        }
        let mut result = vec![0u64; ew]; // FALSE-initialized accumulator
        for e in 0..n {
            let sel = self.index_eq_const(index, e);
            for b in 0..ew {
                let term = self.mk_and(sel, array[e * ew + b]);
                result[b] = self.mk_or(result[b], term);
            }
        }
        result
    }

    /// Array store: produce a new flat array where element `e` becomes
    /// `(index == e) ? value : old_element_e`.
    fn array_write(
        &mut self,
        array: &BvSignal,
        index: &BvSignal,
        value: &BvSignal,
        ew: usize,
        n: usize,
    ) -> BvSignal {
        // Constant address: copy the array and overwrite exactly that element
        // (an out-of-range index is a no-op, matching the all-false selectors).
        if let Some(e) = Self::const_index_value(index) {
            let mut result = array.to_vec();
            if e < n {
                result[e * ew..(e + 1) * ew].copy_from_slice(value);
            }
            return result;
        }
        let mut result = Vec::with_capacity(n * ew);
        for e in 0..n {
            let sel = self.index_eq_const(index, e);
            for b in 0..ew {
                let old = array[e * ew + b];
                result.push(self.mk_mux(sel, value[b], old));
            }
        }
        result
    }

    /// Barrel shifter for a dynamic (variable) shift amount. Processes each
    /// `shamt` bit `k` as a conditional shift by `2^k` (one mux layer), so it
    /// handles any amount — including amounts ≥ width, which shift every bit
    /// out and leave `fill`. `left` selects direction; `fill` is the literal
    /// shifted in (0 for logical, the sign bit for arithmetic right shift).
    /// For a constant `shamt` the per-bit `mk_mux` selectors fold away, so
    /// this reduces to a plain constant shift with no extra gates.
    fn barrel_shift(&mut self, a: &BvSignal, shamt: &BvSignal, left: bool, fill: u64) -> BvSignal {
        let n = a.len();
        let mut cur = a.clone();
        for (k, &sbit) in shamt.iter().enumerate() {
            let shift = 1usize.checked_shl(k as u32).unwrap_or(usize::MAX);
            let mut next = Vec::with_capacity(n);
            for i in 0..n {
                // Bit `i` of `cur` shifted by `shift` in the chosen direction.
                let shifted = if left {
                    if i >= shift {
                        cur[i - shift]
                    } else {
                        fill
                    }
                } else if i.saturating_add(shift) < n {
                    cur[i + shift]
                } else {
                    fill
                };
                next.push(self.mk_mux(sbit, shifted, cur[i]));
            }
            cur = next;
        }
        cur
    }

    /// Get the sort width for a BTOR2 line.
    fn sort_width(&self, program: &Btor2Program, sort_id: i64) -> Result<u32, Btor2Error> {
        match program.sorts.get(&sort_id) {
            Some(Btor2Sort::BitVec(w)) => Ok(*w),
            Some(sort @ Btor2Sort::Array { .. }) => {
                array_flat_width(sort).ok_or_else(|| Btor2Error::ParseError {
                    line: 0,
                    message: "unsupported array sort (non-bitvector index/element) in bit-blasting"
                        .into(),
                })
            }
            None => Err(Btor2Error::ParseError {
                line: 0,
                message: format!("undefined sort {sort_id}"),
            }),
        }
    }

    /// Get the signal for a BTOR2 node reference (may be negated).
    fn get_signal(&self, node_ref: NodeId) -> Result<BvSignal, Btor2Error> {
        let abs_id = node_ref.unsigned_abs() as i64;
        let signal = self
            .signals
            .get(&abs_id)
            .ok_or_else(|| Btor2Error::ParseError {
                line: 0,
                message: format!("signal for node {abs_id} not computed yet"),
            })?;

        if node_ref < 0 {
            // Negation: bitwise NOT of all bits
            Ok(signal.iter().map(|&lit| Self::neg(lit)).collect())
        } else {
            Ok(signal.clone())
        }
    }

    /// Translate the entire BTOR2 program.
    fn translate(&mut self, program: &Btor2Program) -> Result<(), Btor2Error> {
        // First pass: allocate variables for inputs and states.
        for line in &program.lines {
            match &line.node {
                Btor2Node::Input(sort_id, symbol) => {
                    // Array inputs expand to a flat nondeterministic vector whose
                    // aggregate width legitimately exceeds the scalar max_width.
                    let is_array =
                        matches!(program.sorts.get(sort_id), Some(Btor2Sort::Array { .. }));
                    let width = self.sort_width(program, *sort_id)?;
                    if !is_array && width > self.max_width {
                        return Err(Btor2Error::ParseError {
                            line: 0,
                            message: format!("input width {width} exceeds max {}", self.max_width),
                        });
                    }
                    let mut bits = Vec::with_capacity(width as usize);
                    for _ in 0..width {
                        let lit = self.alloc_var();
                        self.inputs.push(lit);
                        bits.push(lit);
                    }
                    let name = symbol
                        .clone()
                        .unwrap_or_else(|| format!("i{}", self.input_bits.len()));
                    self.input_bits.push((name, bits.clone()));
                    self.signals.insert(line.id, bits);
                }
                Btor2Node::State(sort_id, symbol) => {
                    // Array states expand to a flat 2^k*ew-bit latch vector whose
                    // aggregate width legitimately exceeds the scalar max_width;
                    // `bitblast_eligible` already bounded the expansion size.
                    let is_array =
                        matches!(program.sorts.get(sort_id), Some(Btor2Sort::Array { .. }));
                    let width = self.sort_width(program, *sort_id)?;
                    if !is_array && width > self.max_width {
                        return Err(Btor2Error::ParseError {
                            line: 0,
                            message: format!("state width {width} exceeds max {}", self.max_width),
                        });
                    }
                    let mut bits = Vec::with_capacity(width as usize);
                    for _ in 0..width {
                        let lit = self.alloc_var();
                        bits.push(lit);
                    }
                    self.state_lits.insert(line.id, bits.clone());
                    let name = symbol
                        .clone()
                        .unwrap_or_else(|| format!("s{}", self.state_bits.len()));
                    self.state_bits.push((name, bits.clone()));
                    self.signals.insert(line.id, bits);
                }
                _ => {}
            }
        }

        // Second pass: translate operations in source order.
        // We need init/next info collected after computing all signals.
        let mut inits: Vec<(NodeId, NodeId, i64)> = Vec::new(); // (sort_id, state_id, value_ref)
        let mut nexts: Vec<(NodeId, NodeId, i64)> = Vec::new(); // (sort_id, state_id, next_ref)

        for line in &program.lines {
            match &line.node {
                // Skip sorts, inputs, states (handled above).
                Btor2Node::SortBitVec(_)
                | Btor2Node::SortArray(_, _)
                | Btor2Node::Input(_, _)
                | Btor2Node::State(_, _) => {}

                // Init and Next: defer until all signals are computed.
                Btor2Node::Init(sort_id, state_id, value_ref) => {
                    inits.push((*sort_id, *state_id, *value_ref));
                }
                Btor2Node::Next(sort_id, state_id, next_ref) => {
                    nexts.push((*sort_id, *state_id, *next_ref));
                }

                // Constants
                Btor2Node::Zero => {
                    let width = self.sort_width(program, line.sort_id)?;
                    self.signals.insert(line.id, vec![0u64; width as usize]);
                }
                Btor2Node::One => {
                    let width = self.sort_width(program, line.sort_id)?;
                    let mut bits = vec![0u64; width as usize];
                    bits[0] = 1; // LSB = 1
                    self.signals.insert(line.id, bits);
                }
                Btor2Node::Ones => {
                    let width = self.sort_width(program, line.sort_id)?;
                    self.signals.insert(line.id, vec![1u64; width as usize]);
                }
                Btor2Node::Const(s) => {
                    let bits = self.const_from_binary(s);
                    self.signals.insert(line.id, bits);
                }
                Btor2Node::ConstD(s) => {
                    let width = self.sort_width(program, line.sort_id)?;
                    let bits = self.const_from_decimal(s, width)?;
                    self.signals.insert(line.id, bits);
                }
                Btor2Node::ConstH(s) => {
                    let width = self.sort_width(program, line.sort_id)?;
                    let bits = self.const_from_hex(s, width)?;
                    self.signals.insert(line.id, bits);
                }

                // Bitwise unary
                Btor2Node::Not => {
                    let a = self.get_signal(line.args[0])?;
                    let result: BvSignal = a.iter().map(|&lit| Self::neg(lit)).collect();
                    self.signals.insert(line.id, result);
                }
                Btor2Node::Neg => {
                    // Two's complement negation: ~a + 1
                    let a = self.get_signal(line.args[0])?;
                    let not_a: BvSignal = a.iter().map(|&lit| Self::neg(lit)).collect();
                    let one = self.const_one(a.len() as u32);
                    let result = self.add_signals(&not_a, &one);
                    self.signals.insert(line.id, result);
                }
                Btor2Node::Inc => {
                    let a = self.get_signal(line.args[0])?;
                    let one = self.const_one(a.len() as u32);
                    let result = self.add_signals(&a, &one);
                    self.signals.insert(line.id, result);
                }
                Btor2Node::Dec => {
                    let a = self.get_signal(line.args[0])?;
                    let ones = vec![1u64; a.len()]; // all-ones = -1 in two's complement
                    let result = self.add_signals(&a, &ones);
                    self.signals.insert(line.id, result);
                }

                // Reduction ops (produce 1-bit result)
                Btor2Node::Redand => {
                    let a = self.get_signal(line.args[0])?;
                    let result = self.reduce_and(&a);
                    self.signals.insert(line.id, vec![result]);
                }
                Btor2Node::Redor => {
                    let a = self.get_signal(line.args[0])?;
                    let result = self.reduce_or(&a);
                    self.signals.insert(line.id, vec![result]);
                }
                Btor2Node::Redxor => {
                    let a = self.get_signal(line.args[0])?;
                    let result = self.reduce_xor(&a);
                    self.signals.insert(line.id, vec![result]);
                }

                // Bitwise binary ops
                Btor2Node::And => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let result = self.bitwise_and(&a, &b);
                    self.signals.insert(line.id, result);
                }
                Btor2Node::Or => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let result = self.bitwise_or(&a, &b);
                    self.signals.insert(line.id, result);
                }
                Btor2Node::Xor => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let result = self.bitwise_xor(&a, &b);
                    self.signals.insert(line.id, result);
                }
                Btor2Node::Nand => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let and = self.bitwise_and(&a, &b);
                    let result: BvSignal = and.iter().map(|&lit| Self::neg(lit)).collect();
                    self.signals.insert(line.id, result);
                }
                Btor2Node::Nor => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let or = self.bitwise_or(&a, &b);
                    let result: BvSignal = or.iter().map(|&lit| Self::neg(lit)).collect();
                    self.signals.insert(line.id, result);
                }
                Btor2Node::Xnor => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let xor = self.bitwise_xor(&a, &b);
                    let result: BvSignal = xor.iter().map(|&lit| Self::neg(lit)).collect();
                    self.signals.insert(line.id, result);
                }

                // Arithmetic
                Btor2Node::Add => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let result = self.add_signals(&a, &b);
                    self.signals.insert(line.id, result);
                }
                Btor2Node::Sub => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let result = self.sub_signals(&a, &b);
                    self.signals.insert(line.id, result);
                }
                Btor2Node::Mul => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let result = self.mul_signals(&a, &b);
                    self.signals.insert(line.id, result);
                }
                Btor2Node::UDiv => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let (quotient, _rem) = self.divmod_unsigned(&a, &b);
                    self.signals.insert(line.id, quotient);
                }
                Btor2Node::URem => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let (_quot, remainder) = self.divmod_unsigned(&a, &b);
                    self.signals.insert(line.id, remainder);
                }
                Btor2Node::SDiv => {
                    // Truncated signed division: divide magnitudes, sign of the
                    // quotient is sign_a XOR sign_b. The SMT-LIB div-by-zero and
                    // INT_MIN cases fall out of the unsigned divmod + this sign
                    // correction (see divmod_unsigned's div-by-zero rule).
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let w = a.len();
                    let (abs_a, sign_a) = self.abs_signal(&a);
                    let (abs_b, sign_b) = self.abs_signal(&b);
                    let (uq, _ur) = self.divmod_unsigned(&abs_a, &abs_b);
                    let result_sign = self.mk_xor(sign_a, sign_b);
                    let neg_uq = self.negate_signal(&uq);
                    let result: BvSignal = (0..w)
                        .map(|j| self.mk_mux(result_sign, neg_uq[j], uq[j]))
                        .collect();
                    self.signals.insert(line.id, result);
                }
                Btor2Node::SRem => {
                    // Signed remainder: the sign follows the DIVIDEND (`a`).
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let w = a.len();
                    let (abs_a, sign_a) = self.abs_signal(&a);
                    let (abs_b, _sign_b) = self.abs_signal(&b);
                    let (_uq, ur) = self.divmod_unsigned(&abs_a, &abs_b);
                    let neg_ur = self.negate_signal(&ur);
                    let result: BvSignal = (0..w)
                        .map(|j| self.mk_mux(sign_a, neg_ur[j], ur[j]))
                        .collect();
                    self.signals.insert(line.id, result);
                }
                Btor2Node::SMod => {
                    // bvsmod (SMT-LIB): u = |a| urem |b|; the sign follows the
                    // DIVISOR. u==0 ⇒ 0; (sa,sb): (0,0)⇒u, (1,0)⇒-u+b, (0,1)⇒u+b,
                    // (1,1)⇒-u.
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let w = a.len();
                    let (abs_a, sign_a) = self.abs_signal(&a);
                    let (abs_b, sign_b) = self.abs_signal(&b);
                    let (_uq, u) = self.divmod_unsigned(&abs_a, &abs_b);
                    let neg_u = self.negate_signal(&u);
                    let neg_u_plus_b = self.add_signals(&neg_u, &b);
                    let u_plus_b = self.add_signals(&u, &b);
                    // sign_a=1 arm: sign_b ? -u : (-u + b).  sign_a=0 arm: sign_b ? (u+b) : u.
                    let arm_sa1: BvSignal = (0..w)
                        .map(|j| self.mk_mux(sign_b, neg_u[j], neg_u_plus_b[j]))
                        .collect();
                    let arm_sa0: BvSignal = (0..w)
                        .map(|j| self.mk_mux(sign_b, u_plus_b[j], u[j]))
                        .collect();
                    let combined: BvSignal = (0..w)
                        .map(|j| self.mk_mux(sign_a, arm_sa1[j], arm_sa0[j]))
                        .collect();
                    // u == 0 ⇒ result 0.
                    let mut u_nonzero = 0u64;
                    for &bit in &u {
                        u_nonzero = self.mk_or(u_nonzero, bit);
                    }
                    let u_is_zero = Self::neg(u_nonzero);
                    let result: BvSignal = (0..w)
                        .map(|j| self.mk_mux(u_is_zero, 0, combined[j]))
                        .collect();
                    self.signals.insert(line.id, result);
                }
                Btor2Node::Sdivo => {
                    // Signed division overflows ONLY for INT_MIN / -1: a has just
                    // the sign bit set and b is all ones.
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let w = a.len();
                    let mut lower_or = 0u64;
                    for &bit in &a[..w - 1] {
                        lower_or = self.mk_or(lower_or, bit);
                    }
                    let a_is_int_min = self.mk_and(a[w - 1], Self::neg(lower_or));
                    let mut b_all_ones = 1u64;
                    for &bit in &b {
                        b_all_ones = self.mk_and(b_all_ones, bit);
                    }
                    let overflow = self.mk_and(a_is_int_min, b_all_ones);
                    self.signals.insert(line.id, vec![overflow]);
                }
                Btor2Node::Smulo => {
                    // Signed multiply overflow: sign-extend both to 2w, take the
                    // full 2w-bit product, and check the high half is a clean sign
                    // extension of the low w-bit result. overflow iff any bit
                    // prod[w..2w] differs from the result's sign bit prod[w-1].
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let w = a.len();
                    let sign_a = a[w - 1];
                    let sign_b = b[w - 1];
                    let a2: BvSignal = a
                        .iter()
                        .copied()
                        .chain(std::iter::repeat(sign_a).take(w))
                        .collect();
                    let b2: BvSignal = b
                        .iter()
                        .copied()
                        .chain(std::iter::repeat(sign_b).take(w))
                        .collect();
                    let prod = self.mul_signals(&a2, &b2);
                    let low_sign = prod[w - 1];
                    let mut overflow = 0u64;
                    for &hi in &prod[w..(2 * w)] {
                        let diff = self.mk_xor(hi, low_sign);
                        overflow = self.mk_or(overflow, diff);
                    }
                    self.signals.insert(line.id, vec![overflow]);
                }

                // Shifts. A statically constant amount takes the cheap const
                // path; a dynamic amount falls back to a barrel shifter.
                Btor2Node::Sll => {
                    let a = self.get_signal(line.args[0])?;
                    let amount_sig = self.get_signal(line.args[1])?;
                    let result = match self.const_shift_amount(&amount_sig, &line.node) {
                        Ok(amount) => Self::shift_left_const(&a, amount),
                        Err(_) => self.barrel_shift(&a, &amount_sig, true, 0),
                    };
                    self.signals.insert(line.id, result);
                }
                Btor2Node::Srl => {
                    let a = self.get_signal(line.args[0])?;
                    let amount_sig = self.get_signal(line.args[1])?;
                    let result = match self.const_shift_amount(&amount_sig, &line.node) {
                        Ok(amount) => Self::shift_right_logical_const(&a, amount),
                        Err(_) => self.barrel_shift(&a, &amount_sig, false, 0),
                    };
                    self.signals.insert(line.id, result);
                }
                Btor2Node::Sra => {
                    let a = self.get_signal(line.args[0])?;
                    let amount_sig = self.get_signal(line.args[1])?;
                    let result = match self.const_shift_amount(&amount_sig, &line.node) {
                        Ok(amount) => Self::shift_right_arithmetic_const(&a, amount),
                        Err(_) => {
                            // Arithmetic right shift fills with the sign bit.
                            let sign = *a.last().unwrap_or(&0);
                            self.barrel_shift(&a, &amount_sig, false, sign)
                        }
                    };
                    self.signals.insert(line.id, result);
                }

                // Rotates. Eligibility guarantees a CONSTANT amount, so this is a
                // pure bit permutation (no gates): each output bit is wired to an
                // input bit, wrapping around, with the amount taken modulo width.
                Btor2Node::Rol => {
                    let a = self.get_signal(line.args[0])?;
                    let amount_sig = self.get_signal(line.args[1])?;
                    let result = match self.const_shift_amount(&amount_sig, &line.node) {
                        Ok(amount) => Self::rotate_left_const(&a, amount),
                        Err(_) => self.barrel_rotate(&a, &amount_sig, true),
                    };
                    self.signals.insert(line.id, result);
                }
                Btor2Node::Ror => {
                    let a = self.get_signal(line.args[0])?;
                    let amount_sig = self.get_signal(line.args[1])?;
                    let result = match self.const_shift_amount(&amount_sig, &line.node) {
                        Ok(amount) => Self::rotate_right_const(&a, amount),
                        Err(_) => self.barrel_rotate(&a, &amount_sig, false),
                    };
                    self.signals.insert(line.id, result);
                }

                // Unsigned overflow predicates (1-bit results).
                Btor2Node::Uaddo => {
                    // `a + b` overflows (unsigned) iff the ripple-carry adder
                    // emits a final carry-OUT of the MSB.
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let carry = self.add_carry_out(&a, &b);
                    self.signals.insert(line.id, vec![carry]);
                }
                Btor2Node::Usubo => {
                    // `a - b` underflows (unsigned) iff a < b.
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let result = self.ult_signals(&a, &b);
                    self.signals.insert(line.id, vec![result]);
                }
                Btor2Node::Umulo => {
                    // `a * b` overflows (unsigned) iff the DOUBLE-width product
                    // has any bit set above the operand width. Zero-extend both
                    // to 2w, multiply full-width (no truncation loss at 2w), then
                    // OR the high w bits.
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let w = a.len();
                    let a2: BvSignal = a
                        .iter()
                        .copied()
                        .chain(std::iter::repeat(0u64).take(w))
                        .collect();
                    let b2: BvSignal = b
                        .iter()
                        .copied()
                        .chain(std::iter::repeat(0u64).take(w))
                        .collect();
                    let prod = self.mul_signals(&a2, &b2);
                    let mut overflow = 0u64; // FALSE
                    for &hi in &prod[w..(2 * w)] {
                        overflow = self.mk_or(overflow, hi);
                    }
                    self.signals.insert(line.id, vec![overflow]);
                }
                Btor2Node::Saddo => {
                    // Signed add overflow: the operands share a sign but the sum
                    // has the opposite sign. overflow = (sign_a XNOR sign_b) AND
                    // (sign_sum XOR sign_a).
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let w = a.len();
                    let sum = self.add_signals(&a, &b);
                    let sign_a = a[w - 1];
                    let same_sign = Self::neg(self.mk_xor(sign_a, b[w - 1]));
                    let result_flipped = self.mk_xor(sum[w - 1], sign_a);
                    let overflow = self.mk_and(same_sign, result_flipped);
                    self.signals.insert(line.id, vec![overflow]);
                }
                Btor2Node::Ssubo => {
                    // Signed sub overflow: the operands have opposite signs and
                    // the difference has the opposite sign to `a`. overflow =
                    // (sign_a XOR sign_b) AND (sign_diff XOR sign_a).
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let w = a.len();
                    let diff = self.sub_signals(&a, &b);
                    let sign_a = a[w - 1];
                    let diff_signs = self.mk_xor(sign_a, b[w - 1]);
                    let result_flipped = self.mk_xor(diff[w - 1], sign_a);
                    let overflow = self.mk_and(diff_signs, result_flipped);
                    self.signals.insert(line.id, vec![overflow]);
                }

                // Comparison (produce 1-bit result)
                Btor2Node::Eq => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let result = self.eq_signals(&a, &b);
                    self.signals.insert(line.id, vec![result]);
                }
                Btor2Node::Neq => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let eq = self.eq_signals(&a, &b);
                    self.signals.insert(line.id, vec![Self::neg(eq)]);
                }
                Btor2Node::Ult => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let result = self.ult_signals(&a, &b);
                    self.signals.insert(line.id, vec![result]);
                }
                Btor2Node::Ulte => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    // a <= b  iff  !(b < a)
                    let b_lt_a = self.ult_signals(&b, &a);
                    self.signals.insert(line.id, vec![Self::neg(b_lt_a)]);
                }
                Btor2Node::Ugt => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    // a > b  iff  b < a
                    let result = self.ult_signals(&b, &a);
                    self.signals.insert(line.id, vec![result]);
                }
                Btor2Node::Ugte => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    // a >= b  iff  !(a < b)
                    let a_lt_b = self.ult_signals(&a, &b);
                    self.signals.insert(line.id, vec![Self::neg(a_lt_b)]);
                }
                Btor2Node::Slt => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let result = self.slt_signals(&a, &b);
                    self.signals.insert(line.id, vec![result]);
                }
                Btor2Node::Slte => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    // a <= b  iff  !(b < a)
                    let b_lt_a = self.slt_signals(&b, &a);
                    self.signals.insert(line.id, vec![Self::neg(b_lt_a)]);
                }
                Btor2Node::Sgt => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let result = self.slt_signals(&b, &a);
                    self.signals.insert(line.id, vec![result]);
                }
                Btor2Node::Sgte => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    let a_lt_b = self.slt_signals(&a, &b);
                    self.signals.insert(line.id, vec![Self::neg(a_lt_b)]);
                }

                // Boolean/1-bit ops
                Btor2Node::Iff => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    // iff = XNOR for 1-bit
                    let xor = self.bitwise_xor(&a, &b);
                    let result: BvSignal = xor.iter().map(|&lit| Self::neg(lit)).collect();
                    self.signals.insert(line.id, result);
                }
                Btor2Node::Implies => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    // a implies b = !a OR b
                    let not_a: BvSignal = a.iter().map(|&lit| Self::neg(lit)).collect();
                    let result = self.bitwise_or(&not_a, &b);
                    self.signals.insert(line.id, result);
                }

                // Concatenation
                Btor2Node::Concat => {
                    let a = self.get_signal(line.args[0])?;
                    let b = self.get_signal(line.args[1])?;
                    // BTOR2 concat: result = a ## b, where b is lower bits.
                    // Our BvSignal is LSB-first, so result = b ++ a.
                    let mut result = b;
                    result.extend_from_slice(&a);
                    self.signals.insert(line.id, result);
                }

                // Slice
                Btor2Node::Slice(upper, lower) => {
                    let a = self.get_signal(line.args[0])?;
                    let lo = *lower as usize;
                    let hi = *upper as usize;
                    // Fail-closed: a slice whose bounds exceed the operand width
                    // (or are inverted) would panic on `a[lo..=hi]`. Decline.
                    if lo > hi || hi >= a.len() {
                        return Err(Btor2Error::ParseError {
                            line: 0,
                            message: format!(
                                "slice [{hi}:{lo}] out of range for operand of width {}",
                                a.len()
                            ),
                        });
                    }
                    let result = a[lo..=hi].to_vec();
                    self.signals.insert(line.id, result);
                }

                // Extension
                Btor2Node::Uext(extra) => {
                    let a = self.get_signal(line.args[0])?;
                    let mut result = a;
                    // Zero-extend: pad with 0 (FALSE literal) at MSB.
                    result.extend(std::iter::repeat(0u64).take(*extra as usize));
                    self.signals.insert(line.id, result);
                }
                Btor2Node::Sext(extra) => {
                    let a = self.get_signal(line.args[0])?;
                    let sign_bit = *a.last().unwrap_or(&0);
                    let mut result = a;
                    // Sign-extend: replicate MSB.
                    result.extend(std::iter::repeat(sign_bit).take(*extra as usize));
                    self.signals.insert(line.id, result);
                }

                // If-then-else
                Btor2Node::Ite => {
                    let cond = self.get_signal(line.args[0])?;
                    let then_sig = self.get_signal(line.args[1])?;
                    let else_sig = self.get_signal(line.args[2])?;
                    // Fail-closed: an empty condition signal or mismatched
                    // then/else widths would panic on `cond[0]` / `else_sig[i]`.
                    if cond.is_empty() {
                        return Err(Btor2Error::ParseError {
                            line: 0,
                            message: "ite condition signal is empty".into(),
                        });
                    }
                    if then_sig.len() != else_sig.len() {
                        return Err(Btor2Error::ParseError {
                            line: 0,
                            message: format!(
                                "ite branch width mismatch: then={}, else={}",
                                then_sig.len(),
                                else_sig.len()
                            ),
                        });
                    }
                    let sel = cond[0]; // condition is 1-bit
                    let mut result = Vec::with_capacity(then_sig.len());
                    for i in 0..then_sig.len() {
                        result.push(self.mk_mux(sel, then_sig[i], else_sig[i]));
                    }
                    self.signals.insert(line.id, result);
                }

                // Array select: read(array, index) — one-hot mux over elements.
                Btor2Node::Read => {
                    let array = self.get_signal(line.args[0])?;
                    let index = self.get_signal(line.args[1])?;
                    let ew = self.sort_width(program, line.sort_id)? as usize;
                    if ew == 0 || array.len() % ew != 0 {
                        return Err(Btor2Error::ParseError {
                            line: 0,
                            message: format!(
                                "array read width mismatch: array={} bits, element={ew} bits",
                                array.len()
                            ),
                        });
                    }
                    let n = array.len() / ew;
                    let result = self.array_read(&array, &index, ew, n);
                    self.signals.insert(line.id, result);
                }

                // Array store: write(array, index, value) — conditional element update.
                Btor2Node::Write => {
                    let array = self.get_signal(line.args[0])?;
                    let index = self.get_signal(line.args[1])?;
                    let value = self.get_signal(line.args[2])?;
                    let ew = value.len();
                    if ew == 0 || array.len() % ew != 0 {
                        return Err(Btor2Error::ParseError {
                            line: 0,
                            message: format!(
                                "array write width mismatch: array={} bits, value={ew} bits",
                                array.len()
                            ),
                        });
                    }
                    let n = array.len() / ew;
                    let result = self.array_write(&array, &index, &value, ew, n);
                    self.signals.insert(line.id, result);
                }

                // Properties
                Btor2Node::Bad(cond_ref) => {
                    let sig = self.get_signal(*cond_ref)?;
                    // Fail-closed: a bad property must reference a 1-bit signal;
                    // an empty signal would panic on `sig[0]`.
                    let bit = *sig.first().ok_or_else(|| Btor2Error::ParseError {
                        line: 0,
                        message: "bad property references an empty (zero-width) signal".into(),
                    })?;
                    self.bad.push(bit); // bad is 1-bit
                }
                Btor2Node::Constraint(cond_ref) => {
                    let sig = self.get_signal(*cond_ref)?;
                    // Fail-closed: an empty signal would panic on `sig[0]`.
                    let bit = *sig.first().ok_or_else(|| Btor2Error::ParseError {
                        line: 0,
                        message: "constraint references an empty (zero-width) signal".into(),
                    })?;
                    self.constraints.push(bit); // constraint is 1-bit
                }
                Btor2Node::Fair(_) | Btor2Node::Justice(_) | Btor2Node::Output(_) => {
                    // Not needed for safety checking — skip.
                }
            }
        }

        // Process init constraints: set latch reset values.
        //
        // BTOR2 semantics: a `state` node with NO `init` line is UNINITIALIZED —
        // its initial value is UNCONSTRAINED (nondeterministic), NOT zero.
        // Defaulting an un-init latch to 0 is a FALSE-SAFE soundness bug: a net
        // whose property is violated only from a nonzero initial latch value
        // would be wrongly reported SAFE. So we populate `latch_inits` ONLY from
        // explicit `init` lines; a state absent from the map resets
        // nondeterministically (`reset = curr_lit`, which both the AIGER transys
        // and the CHC lowering honor as "no init constraint", and which matches
        // the BTOR2 CHC path that already treats un-init states as unconstrained).
        let state_ids: Vec<NodeId> = self.state_lits.keys().copied().collect();
        let mut latch_inits: HashMap<NodeId, BvSignal> = HashMap::new();

        for (_, state_id, value_ref) in &inits {
            let value_sig = self.get_signal(*value_ref)?;
            latch_inits.insert(*state_id, value_sig);
        }

        // Process next-state functions.
        let mut latch_nexts: HashMap<NodeId, BvSignal> = HashMap::new();
        for (_, state_id, next_ref) in &nexts {
            let next_sig = self.get_signal(*next_ref)?;
            latch_nexts.insert(*state_id, next_sig);
        }

        // Declared sort of each state node. Lets the latch builder recognize an
        // ARRAY state whose flat latch vector is initialized by a SCALAR of the
        // element width — the standard Btor2/BtorMC const-array idiom, where the
        // scalar means "every cell = this scalar".
        let state_sorts: HashMap<NodeId, &Btor2Sort> = program
            .lines
            .iter()
            .filter_map(|line| match &line.node {
                Btor2Node::State(sort_id, _) => program.sorts.get(sort_id).map(|s| (line.id, s)),
                _ => None,
            })
            .collect();

        // Build latch definitions.
        for &state_id in &state_ids {
            let curr_bits = &self.state_lits[&state_id];
            let width = curr_bits.len();

            let next_bits = latch_nexts
                .get(&state_id)
                .cloned()
                .unwrap_or_else(|| curr_bits.clone()); // No next = hold current value.

            // Fail-closed: a next signal narrower than the state width would
            // panic on indexing. Decline on any mismatch. (Unlike init, a
            // scalar `next` of an array is NOT a standard const-array idiom, so
            // we do not broadcast it — a genuine mismatch stays fail-closed.)
            if next_bits.len() != width {
                return Err(Btor2Error::ParseError {
                    line: 0,
                    message: format!(
                        "next-state width {} does not match state width {width} for state {state_id}",
                        next_bits.len()
                    ),
                });
            }

            // Explicit init for this state, resolved to a flat vector matching
            // the state width. Absent ⇒ the latch is uninitialized and resets
            // nondeterministically (see above). Present but the SCALAR element
            // width of an ARRAY state ⇒ const-array broadcast: repeat the scalar
            // across every cell. The flat latch layout is cell-major (cell `c`
            // occupies bits [c*ew, (c+1)*ew), LSB first — see array_read/write),
            // so repeating the scalar's `ew` bits `num` times reproduces the
            // exact reset literals an explicit per-cell init would produce (bit
            // `b` of cell `c` ← scalar bit `b`). Any other width mismatch is a
            // genuine error and stays fail-closed — never silently padded.
            let init_bits: Option<BvSignal> = match latch_inits.get(&state_id) {
                None => None,
                Some(sig) if sig.len() == width => Some(sig.clone()),
                Some(sig) => {
                    let dims = state_sorts.get(&state_id).and_then(|s| array_dims_of(s));
                    match dims {
                        Some((_iw, ew, num, flat))
                            if sig.len() == ew as usize
                                && width == flat as usize
                                && flat as usize == num * ew as usize =>
                        {
                            let mut broadcast = Vec::with_capacity(width);
                            for _ in 0..num {
                                broadcast.extend_from_slice(sig);
                            }
                            debug_assert_eq!(broadcast.len(), width);
                            Some(broadcast)
                        }
                        _ => {
                            return Err(Btor2Error::ParseError {
                                line: 0,
                                message: format!(
                                    "init width {} does not match state width {width} for state {state_id}",
                                    sig.len()
                                ),
                            });
                        }
                    }
                }
            };

            for i in 0..width {
                let curr_lit = curr_bits[i];
                let next_lit = next_bits[i];
                // Reset value: 0 or 1 constant, or curr_lit for nondeterministic.
                let reset = match &init_bits {
                    // No `init` line ⇒ uninitialized ⇒ nondeterministic reset.
                    None => curr_lit,
                    Some(init_bits) => {
                        if init_bits[i] == 0 {
                            0 // Reset to 0
                        } else if init_bits[i] == 1 {
                            1 // Reset to 1
                        } else {
                            // Complex init: this bit of the `init` value blasted to a
                            // GATE literal, i.e. the init value is not a plain constant.
                            // The AIGER latch model cannot express such a reset, and the
                            // historical fallback — a nondeterministic reset — OVER-
                            // approximates the initial-state set by dropping cross-bit
                            // correlations (e.g. `s = concat(i, i)` forces bit0 == bit1).
                            // Over-approximation is only sound for UNSAT verdicts: a
                            // counterexample found from a phantom initial state is
                            // reported as `sat`, a WRONG verdict on a genuinely safe
                            // net. Fail closed instead: `bitblast_eligible` rejects
                            // these nets up front (routing them to the exact word-level
                            // CHC lane), and any caller that still reaches this point
                            // gets a hard error, never a silently over-approximated
                            // circuit.
                            return Err(Btor2Error::ParseError {
                                line: 0,
                                message: format!(
                                    "non-constant init value for state {state_id} (bit {i} \
                                     blasts to a gate literal) reached the bit-blaster; \
                                     bitblast_eligible should have rejected this program"
                                ),
                            });
                        }
                    }
                };
                self.latches.push((curr_lit, next_lit, reset));
            }
        }

        Ok(())
    }

    /// Finish and produce the circuit.
    fn finish(self) -> BitblastedCircuit {
        BitblastedCircuit {
            max_var: self.next_var - 1,
            inputs: self.inputs,
            latches: self.latches,
            ands: self.ands,
            bad: self.bad,
            constraints: self.constraints,
            state_bits: self.state_bits,
            input_bits: self.input_bits,
        }
    }

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    fn const_from_binary(&self, s: &str) -> BvSignal {
        // Binary string is MSB-first. We store LSB-first.
        s.chars().rev().map(|c| u64::from(c == '1')).collect()
    }

    fn const_from_decimal(&self, s: &str, width: u32) -> Result<BvSignal, Btor2Error> {
        let negative = s.starts_with('-');
        let digits = if negative { &s[1..] } else { s };

        // Parse as u128 for handling up to 128-bit constants.
        let abs_val: u128 = digits.parse().map_err(|_| Btor2Error::ParseError {
            line: 0,
            message: format!("invalid decimal constant: {s}"),
        })?;

        let val = if negative {
            // Two's complement: -x = 2^width - x
            let modulus = 1u128 << width;
            modulus.wrapping_sub(abs_val)
        } else {
            abs_val
        };

        let mut bits = Vec::with_capacity(width as usize);
        for i in 0..width {
            bits.push(u64::from((val >> i) & 1 == 1));
        }
        Ok(bits)
    }

    fn const_from_hex(&self, s: &str, width: u32) -> Result<BvSignal, Btor2Error> {
        let val = u128::from_str_radix(s, 16).map_err(|_| Btor2Error::ParseError {
            line: 0,
            message: format!("invalid hex constant: {s}"),
        })?;

        let mut bits = Vec::with_capacity(width as usize);
        for i in 0..width {
            bits.push(u64::from((val >> i) & 1 == 1));
        }
        Ok(bits)
    }

    fn const_one(&self, width: u32) -> BvSignal {
        let mut bits = vec![0u64; width as usize];
        if !bits.is_empty() {
            bits[0] = 1;
        }
        bits
    }

    // -----------------------------------------------------------------------
    // Arithmetic circuits
    // -----------------------------------------------------------------------

    /// Two's-complement negation `-x` = `0 - x`.
    fn negate_signal(&mut self, x: &BvSignal) -> BvSignal {
        let zeros = vec![0u64; x.len()];
        self.sub_signals(&zeros, x)
    }

    /// Absolute value of a two's-complement signal: `(|x|, sign_x)`, where
    /// `|x| = sign_x ? -x : x`. Note `|INT_MIN|` wraps to `INT_MIN`'s bit pattern
    /// (whose UNSIGNED value equals `|INT_MIN|`), which is exactly what the
    /// magnitude divider needs.
    fn abs_signal(&mut self, x: &BvSignal) -> (BvSignal, u64) {
        let w = x.len();
        let sign = x[w - 1];
        let neg = self.negate_signal(x);
        let abs: BvSignal = (0..w).map(|j| self.mk_mux(sign, neg[j], x[j])).collect();
        (abs, sign)
    }

    /// Unsigned restoring division: returns `(quotient, remainder)`, both
    /// `width` bits. Follows SMT-LIB / BTOR2 semantics — division by zero yields
    /// an all-ones quotient and a remainder equal to the dividend (the
    /// `rem >= 0` test is always true when the divisor is 0, so every quotient
    /// bit is set and nothing is subtracted).
    ///
    /// The partial remainder is carried at `width + 1` bits so `rem << 1 | bit`
    /// never overflows (after each step `rem < divisor <= 2^width - 1`, so its
    /// top bit is 0). Uses only the proven ult/sub/shift/mux primitives.
    fn divmod_unsigned(&mut self, a: &BvSignal, b: &BvSignal) -> (BvSignal, BvSignal) {
        let w = a.len();
        if w == 0 {
            return (Vec::new(), Vec::new());
        }
        // Divisor zero-extended to w+1 bits for the wide compare/subtract.
        let b_ext: BvSignal = b.iter().copied().chain(std::iter::once(0u64)).collect();
        let mut rem: BvSignal = vec![0u64; w + 1];
        let mut quot: BvSignal = vec![0u64; w];
        for i in (0..w).rev() {
            // new_rem = (rem << 1) | a[i]  (drops the always-0 top bit of rem).
            let mut new_rem = Self::shift_left_const(&rem, 1);
            new_rem[0] = a[i];
            // ge = new_rem >= b_ext  (unsigned) = NOT(new_rem < b_ext).
            let ge = Self::neg(self.ult_signals(&new_rem, &b_ext));
            let subbed = self.sub_signals(&new_rem, &b_ext);
            // rem = ge ? (new_rem - b) : new_rem;  quot bit i = ge.
            rem = (0..=w)
                .map(|j| self.mk_mux(ge, subbed[j], new_rem[j]))
                .collect();
            quot[i] = ge;
        }
        let remainder: BvSignal = rem[..w].to_vec();
        (quot, remainder)
    }

    /// Shift-add multiplier: `a * b`, truncated to the operand width (BTOR2
    /// `mul` semantics — the low `width` bits, discarding high overflow). For
    /// each bit `i` of `b` the partial product `b_i ? (a << i) : 0` is masked and
    /// accumulated with the ripple-carry adder (both `shift_left_const` and
    /// `add_signals` already truncate to width).
    fn mul_signals(&mut self, a: &BvSignal, b: &BvSignal) -> BvSignal {
        let width = a.len();
        let mut acc = vec![0u64; width]; // 0
        for i in 0..width {
            // Partial product i: (a << i) AND-masked by b_i (a broadcast of b[i]).
            let shifted = Self::shift_left_const(a, i);
            let bi = b[i];
            let masked: BvSignal = shifted.iter().map(|&bit| self.mk_and(bit, bi)).collect();
            acc = self.add_signals(&acc, &masked);
        }
        acc
    }

    /// Ripple-carry: the final carry-OUT of `a + b` — the unsigned
    /// add-overflow bit (`Uaddo`). Unlike [`Self::add_signals`], which discards
    /// the top carry, this propagates the carry through ALL bits and returns it.
    fn add_carry_out(&mut self, a: &BvSignal, b: &BvSignal) -> u64 {
        let width = a.len();
        let mut carry = 0u64; // FALSE
        for i in 0..width {
            // carry_out of bit i = MAJ(a_i, b_i, carry_in) = (a&b) | (carry & (a^b))
            let xor_ab = self.mk_xor(a[i], b[i]);
            let ab = self.mk_and(a[i], b[i]);
            let c_xor = self.mk_and(carry, xor_ab);
            carry = self.mk_or(ab, c_xor);
        }
        carry
    }

    /// Ripple-carry adder. Returns sum (same width as inputs, discards carry).
    fn add_signals(&mut self, a: &BvSignal, b: &BvSignal) -> BvSignal {
        let width = a.len();
        let mut result = Vec::with_capacity(width);
        let mut carry = 0u64; // FALSE

        for i in 0..width {
            // sum_i = a_i XOR b_i XOR carry
            // carry_out = (a_i AND b_i) OR (a_i AND carry) OR (b_i AND carry)
            //           = MAJ(a_i, b_i, carry)
            let xor_ab = self.mk_xor(a[i], b[i]);
            let sum = self.mk_xor(xor_ab, carry);
            result.push(sum);

            if i < width - 1 {
                // carry = (a AND b) OR (carry AND (a XOR b))
                let ab = self.mk_and(a[i], b[i]);
                let c_xor = self.mk_and(carry, xor_ab);
                carry = self.mk_or(ab, c_xor);
            }
        }

        result
    }

    /// Subtraction: a - b = a + (~b) + 1.
    fn sub_signals(&mut self, a: &BvSignal, b: &BvSignal) -> BvSignal {
        let width = a.len();
        let not_b: BvSignal = b.iter().map(|&lit| Self::neg(lit)).collect();

        // Add with carry-in = 1 (for two's complement subtraction).
        let mut result = Vec::with_capacity(width);
        let mut carry = 1u64; // TRUE (the +1)

        for i in 0..width {
            let xor_ab = self.mk_xor(a[i], not_b[i]);
            let sum = self.mk_xor(xor_ab, carry);
            result.push(sum);

            if i < width - 1 {
                let ab = self.mk_and(a[i], not_b[i]);
                let c_xor = self.mk_and(carry, xor_ab);
                carry = self.mk_or(ab, c_xor);
            }
        }

        result
    }

    fn const_shift_amount(
        &self,
        amount_sig: &BvSignal,
        op: &Btor2Node,
    ) -> Result<usize, Btor2Error> {
        bits_to_shift_amount(amount_sig).map_err(|err| Btor2Error::ParseError {
            line: 0,
            message: format!(
                "unsupported dynamic shift amount for {:?} in bit-blasting: {err}",
                op
            ),
        })
    }

    fn shift_left_const(a: &BvSignal, amount: usize) -> BvSignal {
        let width = a.len();
        let mut result = vec![0u64; width];
        if amount >= width {
            return result;
        }
        result[amount..width].copy_from_slice(&a[..(width - amount)]);
        result
    }

    /// Rotate `a` left by a constant `amount` (a bit permutation; the amount is
    /// taken modulo the width, matching BTOR2 `rol` semantics). Signals are
    /// little-endian (bit 0 = LSB), so a left rotate moves each bit toward the
    /// MSB and wraps the top bits back into the low bits:
    /// `result[i] = a[(i - n) mod w]`.
    fn rotate_left_const(a: &BvSignal, amount: usize) -> BvSignal {
        let width = a.len();
        if width == 0 {
            return Vec::new();
        }
        let n = amount % width;
        (0..width).map(|i| a[(i + width - n) % width]).collect()
    }

    /// Rotate `a` right by a constant `amount` (BTOR2 `ror`). Mirror of
    /// [`Self::rotate_left_const`]: `result[i] = a[(i + n) mod w]`.
    fn rotate_right_const(a: &BvSignal, amount: usize) -> BvSignal {
        let width = a.len();
        if width == 0 {
            return Vec::new();
        }
        let n = amount % width;
        (0..width).map(|i| a[(i + n) % width]).collect()
    }

    /// Barrel ROTATOR for a dynamic (variable) rotate amount. Rotation composes
    /// additively mod width, so bit `k` of `amount` conditionally rotates by
    /// `2^k mod width` (one mux layer). The step is doubled mod width each round
    /// to avoid `1 << k` overflow. For a constant amount the mux selectors fold
    /// away, reducing to a plain constant rotate. `left` selects the direction.
    fn barrel_rotate(&mut self, a: &BvSignal, amount: &BvSignal, left: bool) -> BvSignal {
        let w = a.len();
        if w == 0 {
            return Vec::new();
        }
        let mut cur = a.clone();
        let mut step = 1usize % w; // 2^0 mod w
        for &sbit in amount.iter() {
            let rotated = if left {
                Self::rotate_left_const(&cur, step)
            } else {
                Self::rotate_right_const(&cur, step)
            };
            cur = (0..w)
                .map(|i| self.mk_mux(sbit, rotated[i], cur[i]))
                .collect();
            step = (step * 2) % w;
        }
        cur
    }

    fn shift_right_logical_const(a: &BvSignal, amount: usize) -> BvSignal {
        let width = a.len();
        let mut result = vec![0u64; width];
        if amount >= width {
            return result;
        }
        result[..(width - amount)].copy_from_slice(&a[amount..width]);
        result
    }

    fn shift_right_arithmetic_const(a: &BvSignal, amount: usize) -> BvSignal {
        let width = a.len();
        if width == 0 {
            return Vec::new();
        }
        let sign = a[width - 1];
        let mut result = vec![sign; width];
        if amount >= width {
            return result;
        }
        result[..(width - amount)].copy_from_slice(&a[amount..width]);
        result
    }

    // -----------------------------------------------------------------------
    // Comparison circuits
    // -----------------------------------------------------------------------

    /// Equality: all bits must be equal.
    fn eq_signals(&mut self, a: &BvSignal, b: &BvSignal) -> u64 {
        let width = a.len();
        if width == 0 {
            return 1; // vacuously true
        }

        // eq = AND(XNOR(a_i, b_i) for all i)
        let mut eq_bits: Vec<u64> = Vec::with_capacity(width);
        for i in 0..width {
            let xor_i = self.mk_xor(a[i], b[i]);
            eq_bits.push(Self::neg(xor_i)); // XNOR
        }

        self.reduce_and(&eq_bits)
    }

    /// Unsigned less-than: a < b.
    /// Uses subtraction: a < b iff the borrow out of (a - b) is 1,
    /// i.e., the carry out of (a + ~b + 1) is 0.
    fn ult_signals(&mut self, a: &BvSignal, b: &BvSignal) -> u64 {
        let width = a.len();
        if width == 0 {
            return 0; // can't be less than with 0 bits
        }

        let not_b: BvSignal = b.iter().map(|&lit| Self::neg(lit)).collect();

        // Compute carry chain for a + ~b + 1.
        // If final carry = 0, then a < b (borrow occurred).
        let mut carry = 1u64; // +1

        for i in 0..width {
            let xor_ab = self.mk_xor(a[i], not_b[i]);
            let ab = self.mk_and(a[i], not_b[i]);
            let c_xor = self.mk_and(carry, xor_ab);
            carry = self.mk_or(ab, c_xor);
        }

        // a < b iff carry_out == 0 (i.e., borrow)
        Self::neg(carry)
    }

    /// Signed less-than: a < b.
    /// Same as unsigned except the MSB comparison is flipped.
    fn slt_signals(&mut self, a: &BvSignal, b: &BvSignal) -> u64 {
        let width = a.len();
        if width == 0 {
            return 0;
        }
        if width == 1 {
            // 1-bit signed: -1 < 0, so a=1,b=0 means a < b.
            // Signed 1-bit: a < b iff a=1 and b=0
            return self.mk_and(a[0], Self::neg(b[0]));
        }

        // For signed comparison: flip sign bits and do unsigned compare.
        // a_signed < b_signed iff (a with flipped MSB) <_unsigned (b with flipped MSB)
        let mut a_flipped = a.clone();
        let mut b_flipped = b.clone();
        let last = width - 1;
        a_flipped[last] = Self::neg(a[last]);
        b_flipped[last] = Self::neg(b[last]);

        self.ult_signals(&a_flipped, &b_flipped)
    }

    // -----------------------------------------------------------------------
    // Reduction ops
    // -----------------------------------------------------------------------

    fn reduce_and(&mut self, bits: &[u64]) -> u64 {
        if bits.is_empty() {
            return 1; // vacuously true
        }
        let mut result = bits[0];
        for &bit in &bits[1..] {
            result = self.mk_and(result, bit);
        }
        result
    }

    fn reduce_or(&mut self, bits: &[u64]) -> u64 {
        if bits.is_empty() {
            return 0; // vacuously false
        }
        let mut result = bits[0];
        for &bit in &bits[1..] {
            result = self.mk_or(result, bit);
        }
        result
    }

    fn reduce_xor(&mut self, bits: &[u64]) -> u64 {
        if bits.is_empty() {
            return 0;
        }
        let mut result = bits[0];
        for &bit in &bits[1..] {
            result = self.mk_xor(result, bit);
        }
        result
    }

    // -----------------------------------------------------------------------
    // Bitwise binary ops
    // -----------------------------------------------------------------------

    fn bitwise_and(&mut self, a: &BvSignal, b: &BvSignal) -> BvSignal {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| self.mk_and(ai, bi))
            .collect()
    }

    fn bitwise_or(&mut self, a: &BvSignal, b: &BvSignal) -> BvSignal {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| self.mk_or(ai, bi))
            .collect()
    }

    fn bitwise_xor(&mut self, a: &BvSignal, b: &BvSignal) -> BvSignal {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| self.mk_xor(ai, bi))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Combinational circuit evaluator (test support)
// ---------------------------------------------------------------------------

/// Evaluate a bit-blasted circuit COMBINATIONALLY at one step: given boolean
/// values for the inputs (in `circuit.inputs` order) and the current latch bits
/// (in `circuit.latches` order), propagate the AND gates and return
/// `(bad_values, next_latch_values)`.
///
/// The literal encoding is AIGER's: `lit = 2*var + negated`; var 0 is the FALSE
/// constant (so lit 0 = false, lit 1 = true). AND gates are emitted in
/// topological order, so a single forward pass is exact.
///
/// This is the reference oracle for DIFFERENTIAL bit-blast testing — evaluating a
/// circuit against a known truth table (below), and, ahead, checking that two
/// bit-blasts of the same net (e.g. direct array expansion vs Ackermann
/// elimination) agree on `bad` over an enumerated input/state space, WITHOUT a
/// SAT solver (see docs/perf/array-elimination-ackermann-design.md). It is also
/// the combinational evaluator the witness projector ([`crate::witness`]) uses to
/// replay a counterexample frame-by-frame at the word level.
pub(crate) fn lit_val(val: &[bool], lit: u64) -> bool {
    let v = val[(lit >> 1) as usize];
    if lit & 1 == 1 {
        !v
    } else {
        v
    }
}

/// The full variable-value table after one combinational pass at the given
/// input + current-latch assignment (var 0 = FALSE).
pub(crate) fn eval_vals(
    circuit: &BitblastedCircuit,
    input_vals: &[bool],
    latch_vals: &[bool],
) -> Vec<bool> {
    assert_eq!(
        input_vals.len(),
        circuit.inputs.len(),
        "input arity mismatch"
    );
    assert_eq!(
        latch_vals.len(),
        circuit.latches.len(),
        "latch arity mismatch"
    );
    let mut val = vec![false; (circuit.max_var as usize) + 1];
    for (i, &lit) in circuit.inputs.iter().enumerate() {
        val[(lit >> 1) as usize] = input_vals[i];
    }
    for (i, &(curr, _, _)) in circuit.latches.iter().enumerate() {
        val[(curr >> 1) as usize] = latch_vals[i];
    }
    for &(lhs, r0, r1) in &circuit.ands {
        let v = lit_val(&val, r0) && lit_val(&val, r1);
        val[(lhs >> 1) as usize] = v; // lhs is always positive (even)
    }
    val
}

#[cfg(test)]
pub(crate) fn evaluate_circuit(
    circuit: &BitblastedCircuit,
    input_vals: &[bool],
    latch_vals: &[bool],
) -> (Vec<bool>, Vec<bool>) {
    let val = eval_vals(circuit, input_vals, latch_vals);
    let bad = circuit.bad.iter().map(|&l| lit_val(&val, l)).collect();
    let next = circuit
        .latches
        .iter()
        .map(|&(_, nxt, _)| lit_val(&val, nxt))
        .collect();
    (bad, next)
}

/// True iff SOME (input, current-latch) assignment makes a `bad` literal true
/// while ALL `constraint` literals hold. Enumerates the full input+latch space,
/// so tiny nets only. This is the differential-comparison predicate: two
/// bit-blasts of the same net (e.g. direct 2^index array expansion vs Ackermann
/// elimination, whose fresh read vars are extra inputs constrained by
/// consistency) must AGREE on it — the solver-free equisatisfiability check.
#[cfg(test)]
pub(crate) fn bad_reachable(circuit: &BitblastedCircuit) -> bool {
    let ni = circuit.inputs.len();
    let nl = circuit.latches.len();
    let total = ni + nl;
    assert!(
        total <= 22,
        "bad_reachable enumerates 2^(inputs+latches); net too large"
    );
    for mask in 0..(1u64 << total) {
        let inputs: Vec<bool> = (0..ni).map(|i| (mask >> i) & 1 == 1).collect();
        let latches: Vec<bool> = (0..nl).map(|i| (mask >> (ni + i)) & 1 == 1).collect();
        let val = eval_vals(circuit, &inputs, &latches);
        let constraints_ok = circuit.constraints.iter().all(|&c| lit_val(&val, c));
        let any_bad = circuit.bad.iter().any(|&b| lit_val(&val, b));
        if constraints_ok && any_bad {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use crate::types::Btor2Line;

    #[test]
    fn test_bitblast_simple_counter() {
        // 8-bit counter, bad = (count == 0xFF)
        let input = "\
1 sort bitvec 8
2 zero 1
3 state 1 count
4 init 1 3 2
5 one 1
6 add 1 3 5
7 next 1 3 6
8 ones 1
9 eq 1 3 8
10 bad 9
";
        let prog = parse(input).expect("parse");
        let circuit = bitblast(&prog, 32).expect("bitblast");

        // 8 latch bits (one per bit of `count`)
        assert_eq!(circuit.latches.len(), 8);
        // 1 bad property
        assert_eq!(circuit.bad.len(), 1);
        // No inputs
        assert_eq!(circuit.inputs.len(), 0);
        // Should have AND gates for the adder + equality
        assert!(!circuit.ands.is_empty());
    }

    #[test]
    fn test_bitblast_uninitialized_latch_is_nondeterministic_not_zero() {
        // SOUNDNESS regression: a `state` with NO `init` line is UNINITIALIZED,
        // so its reset must be nondeterministic (reset == curr_lit), NOT 0.
        // `bad = s` is reachable only from an initial s == 1; defaulting the
        // latch to 0 would hold it at 0 forever and wrongly report the net SAFE.
        let input = "\
1 sort bitvec 1
2 state 1 s
3 next 1 2 2
4 bad 2
";
        let prog = parse(input).expect("parse");
        let circuit = bitblast(&prog, 32).expect("bitblast");
        assert_eq!(circuit.latches.len(), 1);
        let (curr, _next, reset) = circuit.latches[0];
        assert_eq!(
            reset, curr,
            "uninitialized latch must reset nondeterministically (reset == curr_lit); \
             reset={reset} curr={curr}"
        );
        assert_ne!(
            reset, 0,
            "uninitialized latch must NOT default-reset to 0 (that is the false-safe)"
        );
    }

    #[test]
    fn test_bitblast_explicit_zero_init_still_resets_to_zero() {
        // Contrast: an explicit `init s 0` DOES pin the reset to 0.
        let input = "\
1 sort bitvec 1
2 zero 1
3 state 1 s
4 init 1 3 2
5 next 1 3 3
6 bad 3
";
        let prog = parse(input).expect("parse");
        let circuit = bitblast(&prog, 32).expect("bitblast");
        assert_eq!(circuit.latches.len(), 1);
        let (_curr, _next, reset) = circuit.latches[0];
        assert_eq!(reset, 0, "explicit zero-init must reset to 0");
    }

    #[test]
    fn test_bitblast_const_array_scalar_init_broadcasts_to_every_cell() {
        // A SCALAR init of an ARRAY state is the standard Btor2/BtorMC const-array
        // idiom: "every cell = this scalar". The bit-blaster must BROADCAST the
        // element-width scalar across all cells of the flat latch vector, yielding
        // the SAME reset lits an explicit per-cell init would — never erroring on
        // the element-vs-flat width difference, never silently padding.
        //
        // mem : array[bitvec 1 -> bitvec 8] (2 cells x 8 bits = 16 flat latch
        // bits, cell-major/LSB-first), init'd to `constd 5` = 0b0000_0101. Every
        // cell must reset to that scalar.
        let input = "\
1 sort bitvec 1
2 sort bitvec 8
3 sort array 1 2
4 state 3 mem
5 constd 2 5
6 init 3 4 5
7 next 3 4 4
8 zero 1
9 read 2 4 8
10 eq 1 9 5
11 bad 10
";
        let prog = parse(input).expect("parse");
        let circuit = bitblast(&prog, 32).expect("scalar const-array init bit-blasts");
        // 2 cells x 8 bits ⇒ 16 latch bits.
        assert_eq!(circuit.latches.len(), 16, "2x8 array → 16 latch bits");

        // constd 5 = 0b0000_0101 ⇒ LSB-first bits [1,0,1,0,0,0,0,0], all 0/1 consts.
        let scalar: [u64; 8] = [1, 0, 1, 0, 0, 0, 0, 0];
        let ew = 8usize;
        for cell in 0..2 {
            for b in 0..ew {
                let (_curr, _next, reset) = circuit.latches[cell * ew + b];
                assert_eq!(
                    reset, scalar[b],
                    "cell {cell} bit {b}: broadcast reset must equal scalar bit {b}"
                );
            }
        }
        // Broadcast ⇒ every cell's reset lits are identical (i.e. exactly what an
        // explicit per-cell init to the same scalar would produce).
        let cell0: Vec<u64> = (0..ew).map(|b| circuit.latches[b].2).collect();
        let cell1: Vec<u64> = (0..ew).map(|b| circuit.latches[ew + b].2).collect();
        assert_eq!(
            cell0, cell1,
            "broadcast makes every cell reset to the same lits"
        );
    }

    /// SOUNDNESS trigger net: `init s = concat(i, i)` constrains the initial
    /// state to s ∈ {00, 11}; `next s = s` holds it; `bad = (s == 01)` is
    /// therefore unreachable (true verdict: unsat). The AIGER latch reset is
    /// per-bit constant-or-nondeterministic and cannot express the bit0 ==
    /// bit1 correlation, so the old nondeterministic-reset fallback admitted
    /// the phantom initial states 01/10 and reported a SPURIOUS `sat`.
    const GATE_EXPR_INIT_SAFE_NET: &str = "\
1 sort bitvec 1
2 sort bitvec 2
3 input 1 i
4 state 2 s
5 concat 2 3 3
6 init 2 4 5
7 next 2 4 4
8 constd 2 1
9 eq 1 4 8
10 bad 9
";

    #[test]
    fn test_bitblast_eligible_rejects_gate_expression_init() {
        // SOUNDNESS regression: a gate-expression init must make the program
        // INELIGIBLE for bit-blasting, so the auto-router falls through to the
        // word-level CHC lane, which encodes the init value exactly.
        let prog = parse(GATE_EXPR_INIT_SAFE_NET).expect("parse");
        let err = bitblast_eligible(&prog, 32)
            .expect_err("gate-expression init must be ineligible for bit-blasting");
        assert!(
            err.contains("gate expression"),
            "error should name the gate-expression init, got: {err}"
        );
    }

    #[test]
    fn test_bitblast_gate_expression_init_fails_closed() {
        // Defense in depth: even a caller that skips `bitblast_eligible` must
        // get a hard error from the blaster itself — never a silently over-
        // approximated (nondeterministic-reset) circuit that can report a
        // spurious `sat` on a genuinely safe net.
        let prog = parse(GATE_EXPR_INIT_SAFE_NET).expect("parse");
        let err = bitblast(&prog, 32).expect_err("non-constant init must fail closed");
        let msg = format!("{err}");
        assert!(
            msg.contains("non-constant init"),
            "expected the fail-closed non-constant-init error, got: {msg}"
        );
    }

    #[test]
    fn test_bitblast_eligible_accepts_negated_constant_init() {
        // A NEGATED reference (negative operand id) to a constant node is still
        // a constant — bitwise NOT of constant bits — so it must remain
        // eligible and pin the reset to the negated bits exactly.
        let input = "\
1 sort bitvec 2
2 constd 1 1
3 state 1 s
4 init 1 3 -2
5 next 1 3 3
6 sort bitvec 1
7 redor 6 3
8 bad 7
";
        let prog = parse(input).expect("parse");
        assert!(
            bitblast_eligible(&prog, 32).is_ok(),
            "negated-constant init must stay eligible"
        );
        let circuit = bitblast(&prog, 32).expect("negated-constant init bit-blasts");
        // constd 1 = 0b01; negated = 0b10 ⇒ LSB-first resets [0, 1].
        assert_eq!(circuit.latches.len(), 2);
        assert_eq!(circuit.latches[0].2, 0, "bit 0 resets to NOT(1) = 0");
        assert_eq!(circuit.latches[1].2, 1, "bit 1 resets to NOT(0) = 1");
    }

    #[test]
    fn test_bitblast_array_genuine_width_mismatch_fails_closed() {
        // A scalar init whose width is NEITHER the flat width NOR the element
        // width is a GENUINE mismatch: it must stay fail-closed (declined), never
        // silently broadcast or padded. mem is array[bitvec 1 -> bitvec 8] (flat
        // 16, elem 8); a 4-bit init matches neither ⇒ ParseError.
        let input = "\
1 sort bitvec 1
2 sort bitvec 8
3 sort array 1 2
4 sort bitvec 4
5 state 3 mem
6 constd 4 5
7 init 3 5 6
8 next 3 5 5
9 one 1
10 bad 9
";
        let prog = parse(input).expect("parse");
        let err = bitblast(&prog, 32).expect_err("genuine width mismatch must fail closed");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("init width 4 does not match state width 16"),
            "expected fail-closed width-mismatch error, got: {msg}"
        );
    }

    #[test]
    fn test_bitblast_eligible_rejects_oversized_arrays() {
        // 2^16 elements exceeds the array index-width expansion limit.
        let input = "\
1 sort bitvec 16
2 sort bitvec 32
3 sort array 1 2
4 input 3 mem
5 input 1 addr
6 read 2 4 5
7 bad 6
";
        let prog = parse(input).expect("parse");
        let result = bitblast_eligible(&prog, 32);
        assert!(result.is_err(), "huge array must be rejected");
    }

    #[test]
    fn test_bitblast_array_eligibility_cap_boundary() {
        // Exactly at both caps: index width 12 (= ARRAY_INDEX_MAX_BITS) and
        // 4096 x 2 = 8192 flat bits (= ARRAY_FLAT_MAX_BITS) ⇒ still eligible.
        let at_cap = "\
1 sort bitvec 12
2 sort bitvec 2
3 sort array 1 2
4 input 3 mem
5 input 1 addr
6 read 2 4 5
7 bad 6
";
        assert!(
            bitblast_eligible(&parse(at_cap).expect("parse"), 32).is_ok(),
            "an array exactly at the index+flat caps is eligible"
        );
        // 4096 x 4 = 16384 > 8192: over the flat-bit cap ⇒ declined (→ CHC).
        let over_flat = "\
1 sort bitvec 12
2 sort bitvec 4
3 sort array 1 2
4 input 3 mem
5 input 1 addr
6 read 2 4 5
7 bad 6
";
        assert!(
            bitblast_eligible(&parse(over_flat).expect("parse"), 32).is_err(),
            "an array over the flat-bit cap is declined"
        );
        // Index width 13 > 12: over the index-width cap ⇒ declined regardless.
        let over_index = "\
1 sort bitvec 13
2 sort bitvec 1
3 sort array 1 2
4 input 3 mem
5 input 1 addr
6 read 2 4 5
7 bad 6
";
        assert!(
            bitblast_eligible(&parse(over_index).expect("parse"), 32).is_err(),
            "an array over the index-width cap is declined"
        );
    }

    #[test]
    fn test_array_read_write_constant_index_fast_path() {
        let mut ctx = BitblastContext::new(64);
        // 4 elements x 2 bits (little-endian). e0=[1,0], e1=[0,1], e2=[1,1], e3=[0,0].
        let array: BvSignal = vec![1, 0, 0, 1, 1, 1, 0, 0];
        // Constant-index reads select the element wires directly.
        assert_eq!(ctx.array_read(&array, &vec![0, 1], 2, 4), vec![1, 1]); // index 2
        assert_eq!(ctx.array_read(&array, &vec![0, 0], 2, 4), vec![1, 0]); // index 0
                                                                           // Constant-index write overwrites exactly that element, copies the rest.
        let written = ctx.array_write(&array, &vec![1, 0], &vec![1, 1], 2, 4); // index 1
        assert_eq!(written, vec![1, 0, 1, 1, 1, 1, 0, 0]);
        // Const-index detection: literals 0/1 are constants; a variable (lit 4) is not.
        assert_eq!(BitblastContext::const_index_value(&vec![0, 1]), Some(2));
        assert_eq!(BitblastContext::const_index_value(&vec![4, 0]), None);
    }

    #[test]
    fn test_bitblast_small_array_read_write() {
        // 4-element x 8-bit memory: write v@addr then read back @addr must equal v.
        // bad = (read(write(mem, addr, v), addr) != v) — must be UNSAT (never bad),
        // which exercises array expansion end-to-end through the IC3/PDR portfolio
        // only at the eligibility + bit-blast layer here.
        let input = "\
1 sort bitvec 2
2 sort bitvec 8
3 sort array 1 2
4 state 3 mem
5 input 1 addr
6 input 2 v
7 write 3 4 5 6
8 read 2 7 5
9 sort bitvec 1
10  neq 9 8 6
11 bad 10
";
        let prog = parse(input).expect("parse");
        // A 4x8 array is well within the expansion bounds.
        let max_w = bitblast_eligible(&prog, 32).expect("small array eligible");
        assert_eq!(max_w, 8, "max bitvector width is the element width");
        let circuit = bitblast(&prog, 32).expect("small array bit-blasts");
        // mem expands to 4*8 = 32 latch bits.
        assert_eq!(circuit.latches.len(), 32, "4x8 array → 32 latch bits");
        assert_eq!(circuit.bad.len(), 1);
    }

    #[test]
    fn test_bitblast_eligible_rejects_wide() {
        let input = "\
1 sort bitvec 64
2 input 1 x
3 bad 2
";
        let prog = parse(input).expect("parse");
        let result = bitblast_eligible(&prog, 32);
        assert!(result.is_err());
    }

    #[test]
    fn test_bitblast_1bit_and() {
        let input = "\
1 sort bitvec 1
2 input 1 a
3 input 1 b
4 and 1 2 3
5 bad 4
";
        let prog = parse(input).expect("parse");
        let circuit = bitblast(&prog, 32).expect("bitblast");

        assert_eq!(circuit.inputs.len(), 2);
        assert_eq!(circuit.bad.len(), 1);
        assert_eq!(circuit.latches.len(), 0);
        // AND of two inputs requires 1 AND gate
        assert_eq!(circuit.ands.len(), 1);
    }

    #[test]
    fn test_bitblast_negation() {
        let input = "\
1 sort bitvec 1
2 input 1 x
3 not 1 2
4 bad 3
";
        let prog = parse(input).expect("parse");
        let circuit = bitblast(&prog, 32).expect("bitblast");

        assert_eq!(circuit.inputs.len(), 1);
        assert_eq!(circuit.bad.len(), 1);
        // NOT requires no AND gates (just literal negation)
        assert_eq!(circuit.ands.len(), 0);
    }

    #[test]
    fn test_bitblast_concat_and_slice() {
        let input = "\
1 sort bitvec 4
2 sort bitvec 8
3 sort bitvec 2
4 input 1 lo
5 input 1 hi
6 concat 2 5 4
7 slice 3 6 3 2
8 redor 1 7
9 bad 8
";
        let prog = parse(input).expect("parse");
        let circuit = bitblast(&prog, 32).expect("bitblast");

        // 4+4 = 8 input bits
        assert_eq!(circuit.inputs.len(), 8);
        assert_eq!(circuit.bad.len(), 1);
    }

    #[test]
    fn test_bitblast_ite() {
        let input = "\
1 sort bitvec 1
2 sort bitvec 4
3 input 1 sel
4 input 2 a
5 input 2 b
6 ite 2 3 4 5
7 redor 1 6
8 bad 7
";
        let prog = parse(input).expect("parse");
        let circuit = bitblast(&prog, 32).expect("bitblast");

        // 1 + 4 + 4 = 9 input bits
        assert_eq!(circuit.inputs.len(), 9);
        assert_eq!(circuit.bad.len(), 1);
    }

    #[test]
    fn test_bitblast_constants() {
        let input = "\
1 sort bitvec 4
2 constd 1 10
3 const 1 1010
4 consth 1 a
5 eq 1 2 3
6 eq 1 3 4
7 and 1 5 6
8 bad 7
";
        let prog = parse(input).expect("parse");
        let circuit = bitblast(&prog, 32).expect("bitblast");

        // No inputs, no latches — all constants
        assert_eq!(circuit.inputs.len(), 0);
        assert_eq!(circuit.latches.len(), 0);
        assert_eq!(circuit.bad.len(), 1);
        // bad should be constant TRUE (10==0b1010, 0b1010==0xa).
        assert_eq!(circuit.bad[0], 1);
    }

    #[test]
    fn test_bitblast_constant_shifts() {
        let input = "\
1 sort bitvec 4
2 sort bitvec 1
3 constd 1 3
4 constd 1 1
5 sll 1 3 4
6 constd 1 6
7 eq 2 5 6
8 constd 1 8
9 constd 1 2
10 srl 1 8 9
11 constd 1 2
12 eq 2 10 11
13 const 1 1100
14 sra 1 13 4
15 const 1 1110
16 eq 2 14 15
17 and 2 7 12
18 and 2 17 16
19 bad 18
";
        let prog = parse(input).expect("parse");
        assert_eq!(bitblast_eligible(&prog, 32).expect("eligible"), 4);

        let circuit = bitblast(&prog, 32).expect("bitblast");
        assert_eq!(circuit.inputs.len(), 0);
        assert_eq!(circuit.latches.len(), 0);
        assert_eq!(circuit.bad, vec![1]);
    }

    #[test]
    fn test_bitblast_constant_shift_oversize_amounts() {
        let input = "\
1 sort bitvec 4
2 sort bitvec 1
3 constd 1 7
4 constd 1 4
5 sll 1 3 4
6 zero 1
7 eq 2 5 6
8 const 1 1000
9 sra 1 8 4
10 ones 1
11 eq 2 9 10
12 and 2 7 11
13 bad 12
";
        let prog = parse(input).expect("parse");
        assert_eq!(bitblast_eligible(&prog, 32).expect("eligible"), 4);

        let circuit = bitblast(&prog, 32).expect("bitblast");
        assert_eq!(circuit.bad, vec![1]);
    }

    #[test]
    fn evaluate_circuit_matches_truth_tables() {
        // bad = a AND b (1-bit inputs); exhaustive over the 4 input assignments.
        // (and/or are commutative, so this is robust to input ordering.)
        let and_net = "1 sort bitvec 1\n2 input 1 a\n3 input 1 b\n4 and 1 2 3\n5 bad 4\n";
        let c = bitblast(&parse(and_net).expect("parse"), 32).expect("blast");
        assert_eq!(c.inputs.len(), 2);
        assert!(c.latches.is_empty());
        for a in [false, true] {
            for b in [false, true] {
                let (bad, _next) = evaluate_circuit(&c, &[a, b], &[]);
                assert_eq!(bad, vec![a && b], "and({a},{b})");
            }
        }
        // bad = a OR b.
        let or_net = "1 sort bitvec 1\n2 input 1 a\n3 input 1 b\n4 or 1 2 3\n5 bad 4\n";
        let c = bitblast(&parse(or_net).expect("parse"), 32).expect("blast");
        for a in [false, true] {
            for b in [false, true] {
                let (bad, _next) = evaluate_circuit(&c, &[a, b], &[]);
                assert_eq!(bad, vec![a || b], "or({a},{b})");
            }
        }
        // Const-fold cross-check: uaddo(15,1) overflows 4-bit ⇒ bad const-TRUE,
        // and the evaluator (no inputs/latches) must read it as true.
        let ovf = "1 sort bitvec 4\n2 sort bitvec 1\n3 ones 1\n4 one 1\n5 uaddo 2 3 4\n6 bad 5\n";
        let c = bitblast(&parse(ovf).expect("parse"), 32).expect("blast");
        let (bad, _next) = evaluate_circuit(&c, &[], &[]);
        assert_eq!(bad, vec![true], "15+1 overflow ⇒ bad true");
    }

    #[test]
    fn bad_reachable_detects_and_respects_constraints() {
        // bad = a AND b — reachable (at a=b=1).
        let net = "1 sort bitvec 1\n2 input 1 a\n3 input 1 b\n4 and 1 2 3\n5 bad 4\n";
        let c = bitblast(&parse(net).expect("parse"), 32).expect("blast");
        assert!(bad_reachable(&c), "a∧b is reachable");

        // bad = a AND (NOT a) — never reachable (unsatisfiable).
        let never = "1 sort bitvec 1\n2 input 1 a\n3 not 1 2\n4 and 1 2 3\n5 bad 4\n";
        let c = bitblast(&parse(never).expect("parse"), 32).expect("blast");
        assert!(!bad_reachable(&c), "a∧¬a is unreachable");

        // bad = a, but constraint = NOT a ⇒ every bad-making assignment (a=1)
        // violates the constraint ⇒ unreachable UNDER the constraint. This is
        // exactly the discipline the Ackermann consistency constraints need.
        let constrained = "1 sort bitvec 1\n2 input 1 a\n3 not 1 2\n4 bad 2\n5 constraint 3\n";
        let c = bitblast(&parse(constrained).expect("parse"), 32).expect("blast");
        assert!(!c.constraints.is_empty(), "constraint present");
        assert!(!bad_reachable(&c), "constraint ¬a blocks bad=a");
    }

    #[test]
    fn test_bitblast_dynamic_shift_supported() {
        let input = "\
1 sort bitvec 4
2 sort bitvec 1
3 input 1 value
4 input 1 amount
5 sll 1 3 4
6 redor 2 5
7 bad 6
";
        let prog = parse(input).expect("parse");

        // A dynamic (input-driven) shift amount is now bit-blasted via a barrel
        // shifter rather than rejected.
        let max_w = bitblast_eligible(&prog, 32).expect("dynamic shift is eligible");
        assert_eq!(max_w, 4);

        let circuit = bitblast(&prog, 32).expect("dynamic shift bit-blasts");
        assert_eq!(circuit.bad.len(), 1);
        // value(4 bits) + amount(4 bits) = 8 input literals.
        assert_eq!(circuit.inputs.len(), 8);
    }

    #[test]
    fn test_rotate_const_permutations_are_correct() {
        // Signals are little-endian (bit 0 = LSB). Rotate is a bit permutation.
        // 0b1000 (=[0,0,0,1]) rol 1 = 0b0001 (=[1,0,0,0]).
        assert_eq!(
            BitblastContext::rotate_left_const(&vec![0u64, 0, 0, 1], 1),
            vec![1u64, 0, 0, 0]
        );
        // 0b0001 ror 1 = 0b1000.
        assert_eq!(
            BitblastContext::rotate_right_const(&vec![1u64, 0, 0, 0], 1),
            vec![0u64, 0, 0, 1]
        );
        // 0b0011 rol 1 = 0b0110.
        assert_eq!(
            BitblastContext::rotate_left_const(&vec![1u64, 1, 0, 0], 1),
            vec![0u64, 1, 1, 0]
        );
        // Amount 0 and amount == width are the identity; amount is taken mod w.
        assert_eq!(
            BitblastContext::rotate_left_const(&vec![1u64, 0, 1, 0], 0),
            vec![1u64, 0, 1, 0]
        );
        assert_eq!(
            BitblastContext::rotate_left_const(&vec![1u64, 0, 1, 0], 4),
            vec![1u64, 0, 1, 0]
        );
        assert_eq!(
            BitblastContext::rotate_left_const(&vec![0u64, 0, 0, 1], 5), // 5 mod 4 == 1
            BitblastContext::rotate_left_const(&vec![0u64, 0, 0, 1], 1)
        );
        // rol then ror by the same amount is the identity.
        let a = vec![1u64, 0, 1, 1];
        assert_eq!(
            BitblastContext::rotate_right_const(&BitblastContext::rotate_left_const(&a, 2), 2),
            a
        );
    }

    #[test]
    fn test_bitblast_constant_rotate_is_eligible() {
        // `rol` by a CONSTANT amount is a bit permutation ⇒ eligible + blasts.
        let input = "\
1 sort bitvec 4
2 sort bitvec 1
3 input 1 value
4 one 1
5 rol 1 3 4
6 redor 2 5
7 bad 6
";
        let prog = parse(input).expect("parse");
        let max_w = bitblast_eligible(&prog, 32).expect("constant rotate is eligible");
        assert_eq!(max_w, 4);
        let circuit = bitblast(&prog, 32).expect("constant rotate bit-blasts");
        assert_eq!(circuit.bad.len(), 1);
        assert_eq!(circuit.inputs.len(), 4); // just `value`
    }

    #[test]
    fn test_bitblast_unsigned_overflow_ops_eligible() {
        // Uaddo (carry-out) and Usubo (= a<b) bit-blast to 1-bit predicates.
        let input = "\
1 sort bitvec 4
2 sort bitvec 1
3 input 1 a
4 input 1 b
5 uaddo 2 3 4
6 usubo 2 3 4
7 or 2 5 6
8 bad 7
";
        let prog = parse(input).expect("parse");
        let max_w = bitblast_eligible(&prog, 32).expect("uaddo/usubo are eligible");
        assert_eq!(max_w, 4);
        let circuit = bitblast(&prog, 32).expect("uaddo/usubo bit-blast");
        assert_eq!(circuit.bad.len(), 1);
        assert_eq!(circuit.inputs.len(), 8); // a(4) + b(4)
    }

    #[test]
    fn test_uaddo_constant_operands_fold_to_correct_overflow() {
        // 15 + 1 overflows 4-bit ⇒ the carry-out folds to constant TRUE (lit 1).
        let overflow = "\
1 sort bitvec 4
2 sort bitvec 1
3 ones 1
4 one 1
5 uaddo 2 3 4
6 bad 5
";
        let c = bitblast(&parse(overflow).expect("parse"), 32).expect("blast");
        assert_eq!(c.bad, vec![1], "15+1 overflow ⇒ bad literal is const TRUE");

        // 1 + 1 = 2 fits in 4 bits ⇒ carry-out folds to constant FALSE (lit 0).
        let no_ovf = "\
1 sort bitvec 4
2 sort bitvec 1
3 one 1
4 one 1
5 uaddo 2 3 4
6 bad 5
";
        let c = bitblast(&parse(no_ovf).expect("parse"), 32).expect("blast");
        assert_eq!(
            c.bad,
            vec![0],
            "1+1 no overflow ⇒ bad literal is const FALSE"
        );
    }

    #[test]
    fn test_mul_constant_operands_fold_with_truncation() {
        // 15 * 15 = 225; truncated to 4 bits (mod 16) = 1. Exercises both the
        // shift-add multiplier AND width truncation via constant folding.
        let input = "\
1 sort bitvec 4
2 sort bitvec 1
3 ones 1
4 mul 1 3 3
5 one 1
6 eq 2 4 5
7 bad 6
";
        let c = bitblast(&parse(input).expect("parse"), 32).expect("blast");
        assert_eq!(c.bad, vec![1], "15*15 mod 16 == 1 ⇒ eq folds to const TRUE");
    }

    #[test]
    fn test_umulo_constant_operands_fold() {
        // 15 * 15 = 225 needs 8 bits ⇒ overflows the 4-bit width ⇒ umulo TRUE.
        let ovf = "1 sort bitvec 4\n2 sort bitvec 1\n3 ones 1\n4 umulo 2 3 3\n5 bad 4\n";
        let c = bitblast(&parse(ovf).expect("parse"), 32).expect("blast");
        assert_eq!(c.bad, vec![1], "15*15 overflows 4-bit ⇒ umulo const TRUE");
        // 1 * 1 = 1 fits ⇒ no overflow ⇒ umulo FALSE.
        let no = "1 sort bitvec 4\n2 sort bitvec 1\n3 one 1\n4 umulo 2 3 3\n5 bad 4\n";
        let c = bitblast(&parse(no).expect("parse"), 32).expect("blast");
        assert_eq!(c.bad, vec![0], "1*1 no overflow ⇒ umulo const FALSE");
    }

    #[test]
    fn test_signed_overflow_constant_operands_fold() {
        // saddo: 7 + 1 = 8 overflows signed 4-bit [-8,7] ⇒ TRUE.
        let sadd_ovf =
            "1 sort bitvec 4\n2 sort bitvec 1\n3 const 1 0111\n4 one 1\n5 saddo 2 3 4\n6 bad 5\n";
        let c = bitblast(&parse(sadd_ovf).expect("parse"), 32).expect("blast");
        assert_eq!(c.bad, vec![1], "7+1 signed overflow ⇒ saddo TRUE");
        // saddo: 1 + 1 = 2 fits ⇒ FALSE.
        let sadd_ok =
            "1 sort bitvec 4\n2 sort bitvec 1\n3 one 1\n4 one 1\n5 saddo 2 3 4\n6 bad 5\n";
        let c = bitblast(&parse(sadd_ok).expect("parse"), 32).expect("blast");
        assert_eq!(c.bad, vec![0], "1+1 no signed overflow ⇒ saddo FALSE");
        // ssubo: 7 - (-1) = 8 overflows ⇒ TRUE (a=7, b=ones=-1).
        let ssub_ovf =
            "1 sort bitvec 4\n2 sort bitvec 1\n3 const 1 0111\n4 ones 1\n5 ssubo 2 3 4\n6 bad 5\n";
        let c = bitblast(&parse(ssub_ovf).expect("parse"), 32).expect("blast");
        assert_eq!(c.bad, vec![1], "7-(-1) signed overflow ⇒ ssubo TRUE");
        // ssubo: 3 - 2 = 1 fits ⇒ FALSE.
        let ssub_ok = "1 sort bitvec 4\n2 sort bitvec 1\n3 const 1 0011\n4 const 1 0010\n5 ssubo 2 3 4\n6 bad 5\n";
        let c = bitblast(&parse(ssub_ok).expect("parse"), 32).expect("blast");
        assert_eq!(c.bad, vec![0], "3-2 no signed overflow ⇒ ssubo FALSE");
    }

    #[test]
    fn test_unsigned_divmod_constant_folds() {
        // The result equals the expected constant ⇒ eq folds to a const-TRUE
        // bad literal. Covers exact/inexact division, dividend<divisor, identity,
        // and SMT-LIB div-by-zero (quotient all-ones, remainder = dividend).
        let check = |op: &str, a: &str, b: &str, expected: &str| {
            let net = format!(
                "1 sort bitvec 4\n2 sort bitvec 1\n3 const 1 {a}\n4 const 1 {b}\n\
                 5 {op} 1 3 4\n6 const 1 {expected}\n7 eq 2 5 6\n8 bad 7\n"
            );
            let c = bitblast(&parse(&net).expect("parse"), 32).expect("blast");
            assert_eq!(c.bad, vec![1], "{op} {a}/{b} should fold to {expected}");
        };
        // 15 / 3 = 5 r 0
        check("udiv", "1111", "0011", "0101");
        check("urem", "1111", "0011", "0000");
        // 15 / 4 = 3 r 3
        check("udiv", "1111", "0100", "0011");
        check("urem", "1111", "0100", "0011");
        // 3 / 5 = 0 r 3  (dividend < divisor)
        check("udiv", "0011", "0101", "0000");
        check("urem", "0011", "0101", "0011");
        // 7 / 1 = 7 r 0  (identity)
        check("udiv", "0111", "0001", "0111");
        check("urem", "0111", "0001", "0000");
        // 5 / 0 = 15 (all ones), rem = 5  (SMT-LIB div-by-zero)
        check("udiv", "0101", "0000", "1111");
        check("urem", "0101", "0000", "0101");
    }

    #[test]
    fn test_bitblast_mul_udiv_eligible_sdiv_ineligible() {
        // mul + unsigned udiv/urem bit-blast; SIGNED division stays on CHC.
        let mul = "1 sort bitvec 4\n2 sort bitvec 1\n3 input 1 a\n4 input 1 b\n5 mul 1 3 4\n6 redor 2 5\n7 bad 6\n";
        assert!(
            bitblast_eligible(&parse(mul).expect("parse"), 32).is_ok(),
            "mul is eligible (shift-add multiplier)"
        );
        let udiv = "1 sort bitvec 4\n2 sort bitvec 1\n3 input 1 a\n4 input 1 b\n5 udiv 1 3 4\n6 redor 2 5\n7 bad 6\n";
        assert!(
            bitblast_eligible(&parse(udiv).expect("parse"), 32).is_ok(),
            "udiv is eligible (restoring divider)"
        );
        // Every SCALAR op now bit-blasts (incl. variable rotate). The remaining
        // decline is a bitvector wider than the requested max_width (64 > 32).
        let overwide = "1 sort bitvec 64\n2 sort bitvec 1\n3 input 1 a\n4 input 1 b\n5 add 1 3 4\n6 redor 2 5\n7 bad 6\n";
        assert!(
            bitblast_eligible(&parse(overwide).expect("parse"), 32).is_err(),
            "a bitvector sort wider than max_width stays ineligible"
        );
    }

    #[test]
    fn test_smod_and_sdivo_constant_folds() {
        let check = |op: &str, sort: &str, a: &str, b: &str, expected: &str| {
            let net = format!(
                "1 sort bitvec 4\n2 sort bitvec 1\n3 const 1 {a}\n4 const 1 {b}\n\
                 5 {op} {sort} 3 4\n6 const {sort} {expected}\n7 eq 2 5 6\n8 bad 7\n"
            );
            let c = bitblast(&parse(&net).expect("parse"), 32).expect("blast");
            assert_eq!(c.bad, vec![1], "{op} {a} {b} should fold to {expected}");
        };
        // bvsmod (sign follows the DIVISOR), verified against SMT-LIB QF_BV.
        check("smod", "1", "0101", "0011", "0010"); //  5 smod  3 =  2
        check("smod", "1", "1011", "0011", "0001"); // -5 smod  3 =  1
        check("smod", "1", "0101", "1101", "1111"); //  5 smod -3 = -1
        check("smod", "1", "1011", "1101", "1110"); // -5 smod -3 = -2
        check("smod", "1", "0110", "0011", "0000"); //  6 smod  3 =  0
                                                    // sdivo: overflow only for INT_MIN / -1 (1-bit result, sort 2).
        check("sdivo", "2", "1000", "1111", "1"); // -8 / -1 overflows
        check("sdivo", "2", "0100", "1111", "0"); //  4 / -1 fits
        check("sdivo", "2", "1000", "0010", "0"); // -8 /  2 fits
    }

    #[test]
    fn test_signed_divmod_constant_folds() {
        // Truncated signed division; srem's sign follows the DIVIDEND. Expected
        // values checked against SMT-LIB QF_BV bvsdiv/bvsrem semantics.
        let check = |op: &str, a: &str, b: &str, expected: &str| {
            let net = format!(
                "1 sort bitvec 4\n2 sort bitvec 1\n3 const 1 {a}\n4 const 1 {b}\n\
                 5 {op} 1 3 4\n6 const 1 {expected}\n7 eq 2 5 6\n8 bad 7\n"
            );
            let c = bitblast(&parse(&net).expect("parse"), 32).expect("blast");
            assert_eq!(c.bad, vec![1], "{op} {a} {b} should fold to {expected}");
        };
        check("sdiv", "1010", "0011", "1110"); // -6 / 3  = -2
        check("sdiv", "0110", "1101", "1110"); //  6 / -3 = -2
        check("sdiv", "1010", "1101", "0010"); // -6 / -3 =  2
        check("sdiv", "0111", "0010", "0011"); //  7 / 2  =  3 (trunc)
        check("srem", "1001", "0011", "1111"); // -7 srem 3  = -1 (sign of -7)
        check("srem", "0111", "1101", "0001"); //  7 srem -3 =  1
        check("srem", "1001", "1101", "1111"); // -7 srem -3 = -1
    }

    #[test]
    fn test_smulo_eligible_and_correct() {
        // Signed multiply overflow (smulo) is the LAST scalar op — now bit-blasts.
        let input =
            "1 sort bitvec 4\n2 sort bitvec 1\n3 input 1 a\n4 input 1 b\n5 smulo 2 3 4\n6 bad 5\n";
        let prog = parse(input).expect("parse");
        assert!(
            bitblast_eligible(&prog, 32).is_ok(),
            "smulo now bit-blasts (sign-extended double-width product)"
        );
        // Correctness via const-fold vs signed 4-bit range [-8,7].
        let check = |a: &str, b: &str, expected: &str| {
            let net = format!(
                "1 sort bitvec 4\n2 sort bitvec 1\n3 const 1 {a}\n4 const 1 {b}\n\
                 5 smulo 2 3 4\n6 bad 5\n"
            );
            let c = bitblast(&parse(&net).expect("parse"), 32).expect("blast");
            assert_eq!(
                c.bad,
                vec![if expected == "1" { 1 } else { 0 }],
                "smulo {a}*{b}"
            );
        };
        check("0011", "0011", "1"); //  3 *  3 =  9 overflows
        check("0010", "0011", "0"); //  2 *  3 =  6 fits
        check("1100", "0011", "1"); // -4 *  3 = -12 overflows
        check("1110", "0011", "0"); // -2 *  3 = -6 fits
        check("1000", "1000", "1"); // -8 * -8 =  64 overflows
    }

    #[test]
    fn test_bitblast_variable_rotate_is_eligible() {
        // A VARIABLE (input-driven) rotate amount now bit-blasts via the barrel
        // rotator — it no longer routes to CHC.
        let input = "\
1 sort bitvec 4
2 sort bitvec 1
3 input 1 value
4 input 1 amount
5 rol 1 3 4
6 redor 2 5
7 bad 6
";
        let prog = parse(input).expect("parse");
        assert!(
            bitblast_eligible(&prog, 32).is_ok(),
            "variable-amount rotate is eligible (barrel rotator)"
        );
        let circuit = bitblast(&prog, 32).expect("variable rotate bit-blasts");
        assert_eq!(circuit.bad.len(), 1);
        assert_eq!(circuit.inputs.len(), 8); // value(4) + amount(4)
    }

    #[test]
    fn test_barrel_rotate_matches_const_rotate() {
        // With a constant amount, the barrel rotator's mux selectors fold, so it
        // must reduce to the exact constant rotation — for every amount and both
        // directions. value bits are const literals (1=TRUE, 0=FALSE).
        let value: BvSignal = vec![1, 0, 1, 1];
        for amt in 0..8usize {
            let mut ctx = BitblastContext::new(64);
            let amount: BvSignal = (0..4).map(|k| ((amt >> k) & 1) as u64).collect();
            assert_eq!(
                ctx.barrel_rotate(&value, &amount, true),
                BitblastContext::rotate_left_const(&value, amt),
                "rol by {amt}"
            );
            assert_eq!(
                ctx.barrel_rotate(&value, &amount, false),
                BitblastContext::rotate_right_const(&value, amt),
                "ror by {amt}"
            );
        }
    }

    #[test]
    fn test_bitblast_comparison_ult() {
        let input = "\
1 sort bitvec 4
2 sort bitvec 1
3 input 1 a
4 input 1 b
5 ult 2 3 4
6 bad 5
";
        let prog = parse(input).expect("parse");
        let circuit = bitblast(&prog, 32).expect("bitblast");

        assert_eq!(circuit.inputs.len(), 8); // 4+4 bits
        assert_eq!(circuit.bad.len(), 1);
    }

    #[test]
    fn test_bitblast_constraint() {
        let input = "\
1 sort bitvec 1
2 input 1 x
3 input 1 y
4 and 1 2 3
5 constraint 4
6 bad 2
";
        let prog = parse(input).expect("parse");
        let circuit = bitblast(&prog, 32).expect("bitblast");

        assert_eq!(circuit.constraints.len(), 1);
        assert_eq!(circuit.bad.len(), 1);
    }

    // -- Regression: bug #14 (panic on out-of-range operands) ---------------
    //
    // Slice / ITE / bad / next must DECLINE (ParseError) instead of panicking
    // when bounds exceed the operand width. Programs are built directly so the
    // malformed IR reaches the bit-blaster regardless of parser-level checks.

    #[test]
    fn regression_bitblast_slice_out_of_range_declines() {
        // Operand is 4 bits; slice [10:0] would panic on a[0..=10].
        let mut sorts = HashMap::new();
        sorts.insert(1, Btor2Sort::BitVec(4));
        sorts.insert(2, Btor2Sort::BitVec(11)); // claimed result width
        let lines = vec![
            Btor2Line {
                id: 1,
                sort_id: 0,
                node: Btor2Node::SortBitVec(4),
                args: vec![],
            },
            Btor2Line {
                id: 2,
                sort_id: 1,
                node: Btor2Node::Input(1, Some("a".into())),
                args: vec![],
            },
            Btor2Line {
                id: 3,
                sort_id: 2,
                node: Btor2Node::Slice(10, 0),
                args: vec![2],
            },
            Btor2Line {
                id: 4,
                sort_id: 1,
                node: Btor2Node::Bad(3),
                args: vec![],
            },
        ];
        let prog = Btor2Program {
            lines,
            sorts,
            num_inputs: 1,
            num_states: 0,
            bad_properties: vec![4],
            constraints: vec![],
            fairness: vec![],
            justice: vec![],
        };
        let result = bitblast(&prog, 32);
        assert!(
            matches!(result, Err(Btor2Error::ParseError { .. })),
            "expected ParseError for out-of-range slice, got {result:?}"
        );
    }

    #[test]
    fn regression_bitblast_ite_branch_width_mismatch_declines() {
        // then is 4 bits, else is 2 bits -> would panic on else_sig[i].
        let mut sorts = HashMap::new();
        sorts.insert(1, Btor2Sort::BitVec(1));
        sorts.insert(2, Btor2Sort::BitVec(4));
        sorts.insert(3, Btor2Sort::BitVec(2));
        let lines = vec![
            Btor2Line {
                id: 1,
                sort_id: 0,
                node: Btor2Node::SortBitVec(1),
                args: vec![],
            },
            Btor2Line {
                id: 2,
                sort_id: 0,
                node: Btor2Node::SortBitVec(4),
                args: vec![],
            },
            Btor2Line {
                id: 3,
                sort_id: 0,
                node: Btor2Node::SortBitVec(2),
                args: vec![],
            },
            Btor2Line {
                id: 4,
                sort_id: 1,
                node: Btor2Node::Input(1, Some("sel".into())),
                args: vec![],
            },
            Btor2Line {
                id: 5,
                sort_id: 2,
                node: Btor2Node::Input(2, Some("t".into())),
                args: vec![],
            },
            Btor2Line {
                id: 6,
                sort_id: 3,
                node: Btor2Node::Input(3, Some("e".into())),
                args: vec![],
            },
            Btor2Line {
                id: 7,
                sort_id: 2,
                node: Btor2Node::Ite,
                args: vec![4, 5, 6],
            },
            Btor2Line {
                id: 8,
                sort_id: 1,
                node: Btor2Node::Bad(4),
                args: vec![],
            },
        ];
        let prog = Btor2Program {
            lines,
            sorts,
            num_inputs: 3,
            num_states: 0,
            bad_properties: vec![8],
            constraints: vec![],
            fairness: vec![],
            justice: vec![],
        };
        let result = bitblast(&prog, 32);
        assert!(
            matches!(result, Err(Btor2Error::ParseError { .. })),
            "expected ParseError for ite branch width mismatch, got {result:?}"
        );
    }

    #[test]
    fn regression_bitblast_next_width_mismatch_declines() {
        // State is 4 bits; next value is 2 bits -> would panic on next_bits[i].
        let mut sorts = HashMap::new();
        sorts.insert(1, Btor2Sort::BitVec(4));
        sorts.insert(2, Btor2Sort::BitVec(2));
        sorts.insert(3, Btor2Sort::BitVec(1));
        let lines = vec![
            Btor2Line {
                id: 1,
                sort_id: 0,
                node: Btor2Node::SortBitVec(4),
                args: vec![],
            },
            Btor2Line {
                id: 2,
                sort_id: 0,
                node: Btor2Node::SortBitVec(2),
                args: vec![],
            },
            Btor2Line {
                id: 3,
                sort_id: 0,
                node: Btor2Node::SortBitVec(1),
                args: vec![],
            },
            Btor2Line {
                id: 4,
                sort_id: 1,
                node: Btor2Node::State(1, Some("s".into())),
                args: vec![],
            },
            // 2-bit constant used as the (mismatched) next value.
            Btor2Line {
                id: 5,
                sort_id: 2,
                node: Btor2Node::Zero,
                args: vec![],
            },
            Btor2Line {
                id: 6,
                sort_id: 2,
                node: Btor2Node::Next(2, 4, 5),
                args: vec![],
            },
            Btor2Line {
                id: 7,
                sort_id: 3,
                node: Btor2Node::Input(3, Some("b".into())),
                args: vec![],
            },
            Btor2Line {
                id: 8,
                sort_id: 3,
                node: Btor2Node::Bad(7),
                args: vec![],
            },
        ];
        let prog = Btor2Program {
            lines,
            sorts,
            num_inputs: 1,
            num_states: 1,
            bad_properties: vec![8],
            constraints: vec![],
            fairness: vec![],
            justice: vec![],
        };
        let result = bitblast(&prog, 32);
        assert!(
            matches!(result, Err(Btor2Error::ParseError { .. })),
            "expected ParseError for next-state width mismatch, got {result:?}"
        );
    }

    // -- Regression: bug #17 (re-assert MAX_BV_WIDTH cap in bitblast) -------
    //
    // Even if a caller passes a `max_width` above MAX_BV_WIDTH, the hard 128-bit
    // cap must hold: a wider sort is ineligible / declined. This builds only a
    // tiny IR and never allocates a wide signal.
    #[test]
    fn regression_bitblast_reasserts_max_bv_width() {
        let mut sorts = HashMap::new();
        sorts.insert(1, Btor2Sort::BitVec(200));
        let lines = vec![
            Btor2Line {
                id: 1,
                sort_id: 0,
                node: Btor2Node::SortBitVec(200),
                args: vec![],
            },
            Btor2Line {
                id: 2,
                sort_id: 1,
                node: Btor2Node::Input(1, Some("x".into())),
                args: vec![],
            },
        ];
        let prog = Btor2Program {
            lines,
            sorts,
            num_inputs: 1,
            num_states: 0,
            bad_properties: vec![],
            constraints: vec![],
            fairness: vec![],
            justice: vec![],
        };
        // Caller passes a huge max_width, but the hard cap must still reject.
        assert!(
            bitblast(&prog, u32::MAX).is_err(),
            "bitblast must re-assert MAX_BV_WIDTH even when caller allows more"
        );
        assert!(
            bitblast_eligible(&prog, u32::MAX).is_err(),
            "bitblast_eligible must re-assert MAX_BV_WIDTH even when caller allows more"
        );
    }
}
