//! Error types for TLA+ to ay translation
//!
//! Copyright 2026 Andrew Yates
//! SPDX-License-Identifier: Apache-2.0

use ay_dpll::SolverError;
use thiserror::Error;

/// Errors that can occur during TLA+ to ay translation
#[derive(Debug, Error)]
pub enum AYError {
    /// Variable not found in translation context
    #[error("unknown variable: {0}")]
    UnknownVariable(String),

    /// Type mismatch during translation
    #[error("type mismatch for '{name}': expected {expected}, got {actual}")]
    TypeMismatch {
        /// Variable or expression name
        name: String,
        /// Expected type
        expected: String,
        /// Actual type found
        actual: String,
    },

    /// Expression cannot be translated to ay
    #[error("untranslatable expression: {0}")]
    UntranslatableExpr(String),

    /// Unsupported TLA+ operator
    #[error("unsupported operator: {0}")]
    UnsupportedOp(String),

    /// ay solver returned unknown
    #[error("solver returned unknown")]
    SolverUnknown,

    /// No initial states found
    #[error("no initial states satisfy Init predicate")]
    NoInitStates,

    /// Integer overflow during translation
    #[error("integer too large for SMT: {0}")]
    IntegerOverflow(String),

    /// BMC bound too large
    #[error("BMC bound {bound} exceeds maximum {max}")]
    BmcBoundTooLarge {
        /// The requested BMC bound.
        bound: usize,
        /// The maximum allowed BMC bound (`MAX_BMC_BOUND`).
        max: usize,
    },

    /// ay solver error (sort mismatch, invalid argument, etc.)
    #[error("ay solver error: {0}")]
    Solver(#[from] SolverError),
}

/// Maximum reasonable BMC bound to prevent memory exhaustion.
/// A bound of 1M would allocate 1M ay terms per variable, likely OOM.
pub(crate) const MAX_BMC_BOUND: usize = 100_000;

impl From<String> for AYError {
    fn from(msg: String) -> Self {
        AYError::UntranslatableExpr(msg)
    }
}

/// Result type for ay operations
pub type AYResult<T> = Result<T, AYError>;
