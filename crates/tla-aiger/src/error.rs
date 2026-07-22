// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Error type for AIGER parsing and translation.

use thiserror::Error;

/// An error raised while parsing an AIGER file or translating a circuit.
///
/// Returned by the parser ([`parse_aag`](crate::parse_aag),
/// [`parse_aig`](crate::parse_aig), [`parse_file`](crate::parse_file)) and by
/// CHC translation ([`translate_to_chc`](crate::translate_to_chc),
/// [`check_aiger`](crate::check_aiger)).
#[derive(Debug, Error)]
pub enum AigerError {
    /// A malformed line in the body of an ASCII file (e.g. a non-numeric field).
    #[error("parse error at line {line}: {message}")]
    Parse {
        /// 1-based line number where the error occurred.
        line: usize,
        /// Human-readable description of what was wrong.
        message: String,
    },

    /// The `aag`/`aig` header line is missing, has the wrong tag, or has a
    /// non-numeric M/I/L/O/A/B/C/J/F field.
    #[error("invalid header: {0}")]
    InvalidHeader(String),

    /// The binary input ended mid-record while decoding a delta-encoded AND gate.
    #[error("unexpected EOF while reading binary delta encoding")]
    UnexpectedEof,

    /// A literal references a variable index larger than the declared maxvar.
    #[error("invalid literal {literal}: exceeds maxvar {maxvar}")]
    InvalidLiteral {
        /// The offending literal.
        literal: u64,
        /// The maxvar (`M`) declared in the header.
        maxvar: u64,
    },

    /// A literal references a variable that is never defined as an input, latch,
    /// or AND-gate output.
    #[error("undefined literal {0} referenced")]
    UndefinedLiteral(u64),

    /// The same variable index is defined more than once.
    #[error("duplicate definition for variable {0}")]
    DuplicateDefinition(u64),

    /// An underlying I/O error while reading the file.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The circuit could not be lowered to a CHC problem.
    #[error("translation error: {0}")]
    Translation(String),
}
