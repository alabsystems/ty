// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use thiserror::Error;
use tla_core::Span;

/// Lowering errors for the current supported TIR slice.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TirLowerError {
    /// An AST expression falls outside the currently supported lowering slice.
    #[error("unsupported AST expression `{kind}` at {span:?}")]
    UnsupportedExpr {
        /// Kind name of the unsupported AST expression.
        kind: &'static str,
        /// Source span of the offending expression.
        span: Span,
    },

    /// A span marked as an action subscript did not lower to the canonical
    /// `[A]_v` / `<<A>>_v` shape.
    #[error(
        "invalid action-subscript bridge at {span:?}; marked spans must have canonical lowered `[A]_v` or `<<A>>_v` shape"
    )]
    InvalidActionSubscriptBridge {
        /// Source span of the malformed action subscript.
        span: Span,
    },

    /// A chained INSTANCE/module target was not built from nested module references.
    #[error(
        "invalid chained module target at {span:?}; chained INSTANCE/module targets must be built from nested module references"
    )]
    InvalidChainedTarget {
        /// Source span of the malformed chained target.
        span: Span,
    },

    /// A referenced module or named instance could not be resolved.
    #[error("undefined TLA+ module or named instance `{name}` during TIR lowering at {span:?}")]
    UndefinedModule {
        /// Name of the unresolved module or named instance.
        name: String,
        /// Source span of the reference.
        span: Span,
    },

    /// A referenced operator's body was not available in the lowering environment.
    #[error(
        "undefined operator `{module}!{operator}` during TIR lowering at {span:?}; TIR lowering requires the referenced body to be available in the lowering environment"
    )]
    UndefinedOperator {
        /// Module the operator was expected in.
        module: String,
        /// Name of the unresolved operator.
        operator: String,
        /// Source span of the reference.
        span: Span,
    },

    /// An operator application supplied the wrong number of arguments.
    #[error(
        "arity mismatch for `{module}!{operator}` during TIR lowering at {span:?}: expected {expected}, got {got}"
    )]
    ArityMismatch {
        /// Module the operator is defined in.
        module: String,
        /// Name of the operator.
        operator: String,
        /// Number of parameters the operator declares.
        expected: usize,
        /// Number of arguments supplied at the call site.
        got: usize,
        /// Source span of the application.
        span: Span,
    },

    /// A traversed operator body was expected to be an INSTANCE definition but was not.
    #[error(
        "operator body for `{module}!{operator}` is not an INSTANCE definition at {span:?}; INSTANCE-chain lowering only traverses operators whose bodies resolve to INSTANCE expressions"
    )]
    ExpectedInstance {
        /// Module the operator is defined in.
        module: String,
        /// Name of the operator whose body was not an INSTANCE.
        operator: String,
        /// Source span of the reference.
        span: Span,
    },

    /// A cycle was detected while resolving INSTANCE/module references.
    #[error(
        "recursive INSTANCE/module reference `{module}!{operator}` encountered during TIR lowering at {span:?}"
    )]
    RecursiveModuleRef {
        /// Module participating in the reference cycle.
        module: String,
        /// Operator participating in the reference cycle.
        operator: String,
        /// Source span where the cycle was detected.
        span: Span,
    },

    /// `@` (EXCEPT-AT) appeared outside the value subtree of an `EXCEPT`.
    #[error(
        "`@` (EXCEPT-AT) used outside an EXCEPT value subtree at {span:?}; `@` is only valid inside EXCEPT right-hand-side expressions"
    )]
    InvalidExceptAt {
        /// Source span of the misplaced `@`.
        span: Span,
    },
}
