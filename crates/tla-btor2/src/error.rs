// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

// Author: Andrew Yates

//! Error type and width bound for BTOR2 parsing, validation, and translation.
//!
//! Every fallible operation in this crate returns [`Btor2Error`]. Variants
//! distinguish I/O failures, syntactic [`Btor2Error::ParseError`]s, and the
//! semantic cross-reference failures caught during validation (undefined nodes,
//! bad sort references, duplicate ids, wrong operand counts). The
//! [`MAX_BV_WIDTH`] constant is the hard cap that keeps the `u128`-backed value
//! encoding fail-closed: an over-wide bitvector is rejected up front rather than
//! silently truncated.

/// Unique identifier for a BTOR2 line/node.
pub type NodeId = i64;

/// Maximum supported bitvector width, in bits.
///
/// Every default translation path (CHC via `to_chc`/`translate`, and
/// bit-blasting) materializes bitvector constants and arithmetic through a
/// `u128`. A width greater than 128 cannot be represented faithfully: a wide
/// constant would silently truncate to its low 128 bits (or to 0), and
/// arithmetic would be masked mod 2^128, producing a SAFE/UNSAFE verdict over
/// the WRONG model. To stay fail-closed we reject any width above this bound up
/// front, before any value is materialized.
pub const MAX_BV_WIDTH: u32 = 128;

/// Error type for BTOR2 parsing and validation.
#[derive(Debug, thiserror::Error)]
pub enum Btor2Error {
    /// I/O error when reading a BTOR2 file.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Parse error at a specific line.
    #[error("parse error at line {line}: {message}")]
    ParseError {
        /// 1-based source line number where the error was detected.
        line: usize,
        /// Human-readable description of the malformed input.
        message: String,
    },

    /// Reference to an invalid or undefined sort.
    #[error("invalid sort id {sort_id} at line {line}")]
    InvalidSort {
        /// 1-based source line number making the reference.
        line: usize,
        /// The sort id that was referenced but not defined.
        sort_id: NodeId,
    },

    /// Reference to an undefined node.
    #[error("undefined node id {node_id} at line {line}")]
    UndefinedNode {
        /// 1-based source line number making the reference.
        line: usize,
        /// The node id that was referenced but not defined.
        node_id: NodeId,
    },

    /// A sort ID does not reference a `sort` node.
    #[error("node {node_id} at line {line} is not a sort")]
    NotASort {
        /// 1-based source line number making the reference.
        line: usize,
        /// The node id used where a `sort` node was required.
        node_id: NodeId,
    },

    /// Duplicate node ID.
    #[error("duplicate node id {node_id} at line {line}")]
    DuplicateId {
        /// 1-based source line number of the redefinition.
        line: usize,
        /// The node id that was already defined earlier.
        node_id: NodeId,
    },

    /// Wrong number of arguments for an operator.
    #[error("invalid argument count for {op} at line {line}: expected {expected}, got {got}")]
    InvalidArgCount {
        /// 1-based source line number of the operator.
        line: usize,
        /// The operator mnemonic (e.g. `"add"`, `"ite"`).
        op: String,
        /// Number of operands the operator requires.
        expected: usize,
        /// Number of operands actually supplied.
        got: usize,
    },
}
