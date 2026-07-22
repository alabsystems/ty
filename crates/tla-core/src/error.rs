// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Error types for tla-core

use crate::span::Span;
use thiserror::Error;

/// Result type for tla-core operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during parsing or analysis
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// The source could not be parsed (malformed TLA+ syntax).
    #[error("Syntax error: {message}")]
    Syntax {
        /// Human-readable description of the syntax problem.
        message: String,
        /// Location of the offending token(s).
        span: Span,
    },

    /// A name was referenced but no definition is in scope.
    #[error("Undefined name: {name}")]
    UndefinedName {
        /// The unresolved identifier.
        name: String,
        /// Location of the reference.
        span: Span,
    },

    /// Two definitions in the same scope share a name.
    #[error("Duplicate definition: {name}")]
    DuplicateDefinition {
        /// The conflicting name.
        name: String,
        /// Location of the first (winning) definition.
        original: Span,
        /// Location of the conflicting redefinition.
        duplicate: Span,
    },

    /// An expression is ill-typed for its context.
    #[error("Type error: {message}")]
    Type {
        /// Human-readable description of the type mismatch.
        message: String,
        /// Location of the ill-typed expression.
        span: Span,
    },

    /// An operator was applied to the wrong number of arguments.
    #[error("Arity mismatch: expected {expected} arguments, got {got}")]
    ArityMismatch {
        /// Number of arguments the operator declares.
        expected: usize,
        /// Number of arguments actually supplied.
        got: usize,
        /// Location of the application.
        span: Span,
    },

    /// An `EXTENDS`/`INSTANCE` referenced a module that could not be located.
    #[error("Module not found: {name}")]
    ModuleNotFound {
        /// The missing module's name.
        name: String,
        /// Location of the reference.
        span: Span,
    },

    /// An I/O error occurred while reading a source file.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
