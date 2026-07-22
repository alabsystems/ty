// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! AY-backed Init state enumeration using ALL-SAT with blocking clauses.
//!
//! Supports finite membership domains, arithmetic constraints, and quantified
//! finite domains that benefit from symbolic solving over brute-force search.
//! Part of #251.

use std::sync::Arc;

mod model;
mod type_inference;
mod value_convert;

#[cfg(feature = "ay")]
mod enumerate;

#[cfg(feature = "ay")]
pub(crate) use enumerate::enumerate_init_states_ay;

#[cfg(all(test, feature = "ay"))]
pub(crate) use enumerate::ensure_init_enumeration_sat_profile_accepted;

#[cfg(all(test, feature = "ay"))]
use self::type_inference::{infer_sort_from_expr, infer_sort_from_set};

pub(crate) use value_convert::extract_var_name;

/// Result type for ay enumeration operations
pub(crate) type AYEnumResult<T> = Result<T, AYEnumError>;

/// Errors that can occur during ay-based enumeration
#[derive(Debug)]
pub(crate) enum AYEnumError {
    /// Failed to translate Init predicate to ay
    TranslationFailed(String),
    /// ay solver returned unknown (unclassified or incomplete)
    SolverUnknown,
    /// ay solver timed out (Part of #2826)
    SolverTimeout,
    /// ay solver panicked or produced a typed failure (Part of #2826)
    SolverFailed(String),
    /// Exceeded maximum number of solutions
    MaxSolutionsExceeded(usize),
    /// Variable type not supported for ay translation
    UnsupportedVarType { var: String, reason: String },
    /// ay returned invalid model
    InvalidModel(String),
}

impl std::fmt::Display for AYEnumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AYEnumError::TranslationFailed(msg) => {
                write!(f, "failed to translate Init to ay: {}", msg)
            }
            AYEnumError::SolverUnknown => write!(f, "ay solver returned unknown"),
            AYEnumError::SolverTimeout => write!(f, "ay solver timed out"),
            AYEnumError::SolverFailed(msg) => {
                write!(f, "ay solver failed: {}", msg)
            }
            AYEnumError::MaxSolutionsExceeded(n) => {
                write!(f, "exceeded maximum solutions limit ({})", n)
            }
            AYEnumError::UnsupportedVarType { var, reason } => {
                write!(f, "variable '{}' not supported for ay: {}", var, reason)
            }
            AYEnumError::InvalidModel(msg) => write!(f, "invalid ay model: {}", msg),
        }
    }
}

impl std::error::Error for AYEnumError {}

/// Configuration for ay-based Init enumeration
#[derive(Debug, Clone)]
pub(crate) struct AYEnumConfig {
    /// Maximum number of solutions to enumerate (default: 1_000_000)
    pub(crate) max_solutions: usize,
    /// Timeout for each solver `check_sat` call (Part of #2826).
    /// `None` means no timeout (default).
    pub(crate) solve_timeout: Option<std::time::Duration>,
    /// Enable debug output
    pub(crate) debug: bool,
}

impl Default for AYEnumConfig {
    fn default() -> Self {
        AYEnumConfig {
            max_solutions: 1_000_000,
            solve_timeout: None,
            debug: debug_ay_enabled(),
        }
    }
}

debug_flag!(debug_ay_enabled, "TY_DEBUG_AY");

/// Information about a state variable for ay translation
#[derive(Debug, Clone)]
pub(crate) struct VarInfo {
    /// Variable name
    pub(crate) name: Arc<str>,
    /// Inferred sort (Int, Bool, or Function)
    pub(crate) sort: VarSort,
}

/// Variable sort for ay translation
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum VarSort {
    /// Boolean variable
    Bool,
    /// Integer variable
    Int,
    /// String variable with finite domain
    String {
        /// Finite domain of string values
        domain: Vec<String>,
    },
    /// Function with finite domain
    Function {
        /// Domain element keys
        domain_keys: Vec<String>,
        /// Range sort
        range: Box<VarSort>,
    },
    /// Tuple with fixed element sorts (1-indexed)
    Tuple {
        /// Sort of each element (index 0 = element 1, etc.)
        element_sorts: Vec<VarSort>,
    },
    /// Heterogeneous set - cannot be represented in ay
    /// This is set when SetEnum contains mixed types (e.g., {1, "a"})
    /// Part of #523: soundness fix for heterogeneous SetEnum
    Heterogeneous {
        /// Description of why this is heterogeneous
        reason: String,
    },
}

#[cfg(test)]
#[cfg(feature = "ay")]
mod tests;
