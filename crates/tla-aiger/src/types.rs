// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Core types for the AIGER And-Inverter Graph format.
//!
//! An [`AigerCircuit`] is the parsed in-memory representation of a `.aag`/`.aig`
//! file: a flat list of inputs, latches, AND gates, and properties over a single
//! [`Literal`] address space. Literals are `u64` where the low bit is the sign
//! and `lit >> 1` is the variable index; literal `0` is constant FALSE and `1`
//! is constant TRUE. The free functions ([`aiger_var`], [`aiger_not`], etc.)
//! manipulate this encoding.
//!
//! Reference: "The AIGER And-Inverter Graph (AIG) Format Version 20071012"
//! by Armin Biere, Johannes Kepler University, 2006-2007.
//! Extended with BCJF sections for HWMCC model checking competitions.

/// A literal in an AIGER circuit. The LSB is the sign bit (1 = negated).
/// Variable index = literal >> 1. Literal 0 = FALSE, 1 = TRUE.
pub type Literal = u64;

/// Strip the sign bit from a literal, returning the positive (even) form.
#[inline]
pub fn aiger_strip(lit: Literal) -> Literal {
    lit & !1
}

/// Get the variable index from a literal.
#[inline]
pub fn aiger_var(lit: Literal) -> u64 {
    lit >> 1
}

/// Convert a variable index to its positive literal.
#[inline]
pub fn aiger_var2lit(var: u64) -> Literal {
    var << 1
}

/// Check if a literal is negated.
#[inline]
pub fn aiger_is_negated(lit: Literal) -> bool {
    lit & 1 == 1
}

/// Negate a literal (toggle the sign bit).
#[inline]
pub fn aiger_not(lit: Literal) -> Literal {
    lit ^ 1
}

/// An AND gate: lhs = rhs0 AND rhs1. LHS is always even (positive literal).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AigerAnd {
    /// Output literal of the gate. Always even (positive); it defines a variable.
    pub lhs: Literal,
    /// First (possibly negated) input literal.
    pub rhs0: Literal,
    /// Second (possibly negated) input literal.
    pub rhs1: Literal,
}

/// A latch (sequential state element).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AigerLatch {
    /// Current-state literal (always even/positive).
    pub lit: Literal,
    /// Next-state literal.
    pub next: Literal,
    /// Reset value: 0, 1, or equal to `lit` (meaning undefined/nondeterministic).
    pub reset: Literal,
    /// Optional symbolic name.
    pub name: Option<String>,
}

/// A named symbol (for inputs, outputs, bad, constraints, justice, fairness).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AigerSymbol {
    /// The literal this symbol refers to.
    pub lit: Literal,
    /// Optional symbolic name from the file's symbol table.
    pub name: Option<String>,
}

/// A justice property: a set of fairness conditions that must all hold
/// infinitely often.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AigerJustice {
    /// The literals whose conjunction must hold infinitely often.
    pub lits: Vec<Literal>,
}

/// A complete parsed AIGER circuit.
#[derive(Debug, Clone)]
pub struct AigerCircuit {
    /// Maximum variable index (M in the header).
    pub maxvar: u64,

    /// Input literals (always even/positive). Count = I.
    pub inputs: Vec<AigerSymbol>,

    /// Latches with next-state functions. Count = L.
    pub latches: Vec<AigerLatch>,

    /// Output literals. Count = O.
    pub outputs: Vec<AigerSymbol>,

    /// AND gates. Count = A.
    pub ands: Vec<AigerAnd>,

    /// Bad-state property literals. Count = B (extended header).
    pub bad: Vec<AigerSymbol>,

    /// Constraint (assumption) literals. Count = C (extended header).
    pub constraints: Vec<AigerSymbol>,

    /// Justice properties (each is a set of fairness literals). Count = J.
    pub justice: Vec<AigerJustice>,

    /// Fairness constraint literals. Count = F.
    pub fairness: Vec<AigerSymbol>,

    /// Comment lines (after the 'c' marker).
    pub comments: Vec<String>,
}

impl AigerCircuit {
    /// Returns the total number of defined variables (inputs + latches + ands).
    pub fn num_defined(&self) -> usize {
        self.inputs.len() + self.latches.len() + self.ands.len()
    }

    /// Returns true if this circuit has any safety properties to check.
    pub fn has_safety_properties(&self) -> bool {
        !self.bad.is_empty() || !self.outputs.is_empty()
    }

    /// Returns true if this circuit has liveness properties.
    pub fn has_liveness_properties(&self) -> bool {
        !self.justice.is_empty() || !self.fairness.is_empty()
    }
}
