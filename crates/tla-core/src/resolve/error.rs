// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Error types for name resolution.

use super::types::SymbolKind;
use crate::span::Span;

/// Error during name resolution
#[derive(Debug, Clone)]
pub struct ResolveError {
    /// Error kind
    pub kind: ResolveErrorKind,
    /// Span where error occurred
    pub span: Span,
}

/// Kinds of resolution errors
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ResolveErrorKind {
    /// Reference to undefined identifier
    Undefined {
        /// The unresolved identifier.
        name: String,
    },
    /// Duplicate definition in same scope
    Duplicate {
        /// The redefined name.
        name: String,
        /// Location of the first (winning) definition.
        first_def: Span,
    },
    /// Conflicting operator arity between imported definitions.
    ///
    /// TLC treats same-kind duplicate symbols as a warning only if their arity matches; otherwise,
    /// it is an error.
    ImportedOperatorArityConflict {
        /// The operator name defined with differing arities.
        name: String,
        /// Location of the first imported definition.
        first_def: Span,
        /// Arity of the first imported definition.
        first_arity: usize,
        /// Arity of the conflicting second definition.
        second_arity: usize,
    },
    /// Wrong arity in operator application
    ArityMismatch {
        /// The applied operator's name.
        name: String,
        /// Number of parameters the operator declares.
        expected: usize,
        /// Number of arguments supplied at the call site.
        got: usize,
    },
    /// Using variable where operator expected (or vice versa)
    KindMismatch {
        /// The misused name.
        name: String,
        /// The symbol kind required by the use site.
        expected: SymbolKind,
        /// The symbol kind the name was actually declared with.
        got: SymbolKind,
    },
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            ResolveErrorKind::Undefined { name } => {
                write!(f, "undefined identifier `{name}`")
            }
            ResolveErrorKind::Duplicate { name, .. } => {
                write!(f, "duplicate definition of `{name}`")
            }
            ResolveErrorKind::ImportedOperatorArityConflict {
                name,
                first_arity,
                second_arity,
                ..
            } => {
                write!(
                    f,
                    "conflicting definitions of operator `{name}` (arity {first_arity} vs {second_arity})"
                )
            }
            ResolveErrorKind::ArityMismatch {
                name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "operator `{name}` expects {expected} arguments, got {got}"
                )
            }
            ResolveErrorKind::KindMismatch {
                name,
                expected,
                got,
            } => {
                write!(f, "`{name}` is a {got:?}, expected {expected:?}")
            }
        }
    }
}

impl std::error::Error for ResolveError {}
