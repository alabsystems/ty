// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Evaluation errors for the TLA+ model checker.
//!
//! Value-dependent convenience constructors (type_error, one_argument_error,
//! evaluating_error, argument_error) are defined in `value/error_constructors.rs`
//! to break the error->value circular dependency.
//!
//! Part of #1269 Phase 2: break the error<->value cycle.

use thiserror::Error;
use tla_core::Span;

mod messages;
#[cfg(test)]
mod tests;

use self::messages::{
    _format_argument_error, _format_index_out_of_bounds, _format_no_such_field,
    _format_not_in_domain, _format_one_argument_error,
};

/// Evaluation error
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum EvalError {
    /// Type mismatch in operation
    #[error("Type error: expected {expected}, got {got}")]
    TypeError {
        /// Human-readable description of the type the operation required.
        expected: &'static str,
        /// Human-readable description of the type actually supplied.
        got: &'static str,
        /// Source location of the offending expression, if known.
        span: Option<Span>,
    },

    /// Division by zero (TLC: EC.TLC_MODULE_DIVISION_BY_ZERO)
    #[error("The second argument of \\div is 0.")]
    DivisionByZero {
        /// Source location of the offending `\div` expression, if known.
        span: Option<Span>,
    },

    /// Modulus operator requires positive divisor (TLC: EC.TLC_MODULE_ARGUMENT_ERROR)
    #[error("The second argument of % should be a positive number, but instead it is:\n{value}")]
    ModulusNotPositive {
        /// Rendered value of the non-positive divisor.
        value: String,
        /// Source location of the offending `%` expression, if known.
        span: Option<Span>,
    },

    /// Undefined variable reference
    #[error("Undefined variable: {name}")]
    UndefinedVar {
        /// Name of the variable that could not be resolved.
        name: String,
        /// Source location of the reference, if known.
        span: Option<Span>,
    },

    /// Undefined operator reference
    #[error("Undefined operator: {name}")]
    UndefinedOp {
        /// Name of the operator that could not be resolved.
        name: String,
        /// Source location of the reference, if known.
        span: Option<Span>,
    },

    /// Function applied to value not in domain (TLC: FcnRcdValue.java:354)
    /// When func_display is provided, uses TLC-compatible verbose format.
    #[error("{}", _format_not_in_domain(.arg, .func_display))]
    NotInDomain {
        /// Rendered argument that fell outside the function's domain.
        arg: String,
        /// Optional rendered function, enabling TLC's verbose message form.
        func_display: Option<String>,
        /// Source location of the offending application, if known.
        span: Option<Span>,
    },

    /// Record field not found (TLC: RecordValue.java:488)
    /// With record_display: "Attempted to access nonexistent field '<field>' of record\n<record>"
    /// Without: "Record has no field: <field>"
    #[error("{}", _format_no_such_field(.field, .record_display))]
    NoSuchField {
        /// Name of the field that does not exist on the record.
        field: String,
        /// Optional rendered record, enabling TLC's verbose message form.
        record_display: Option<String>,
        /// Source location of the offending access, if known.
        span: Option<Span>,
    },

    /// Sequence/tuple index out of bounds (TLC: TupleValue.java:144)
    /// With value_display: "Attempted to access index <N> of tuple\n<tuple>\nwhich is out of bounds."
    /// Without: "Sequence index out of bounds: <index> not in 1..<len>"
    #[error("{}", _format_index_out_of_bounds(.index, .len, .value_display))]
    IndexOutOfBounds {
        /// The 1-based index that was requested.
        index: i64,
        /// Length of the sequence/tuple being indexed.
        len: usize,
        /// Optional rendered value, enabling TLC's verbose message form.
        value_display: Option<String>,
        /// Source location of the offending access, if known.
        span: Option<Span>,
    },

    /// TLC-compatible single-argument type error (TLC: EC.TLC_MODULE_ONE_ARGUMENT_ERROR = 2283)
    /// Format: "The argument of <op> should be a <type>, but instead it is:\n<value>"
    /// Used by Len, Head, Tail.
    #[error("{}", _format_one_argument_error(.op, .expected_type, .value_display))]
    OneArgumentError {
        /// Name of the operator that rejected its argument (e.g. `Len`).
        op: &'static str,
        /// Human-readable description of the type the operator required.
        expected_type: &'static str,
        /// Rendered value of the rejected argument.
        value_display: String,
        /// Source location of the offending application, if known.
        span: Option<Span>,
    },

    /// TLC-compatible empty sequence error (TLC: EC.TLC_MODULE_APPLY_EMPTY_SEQ = 2184)
    /// Format: "Attempted to apply <op> to the empty sequence."
    /// Used by Head, Tail.
    #[error("Attempted to apply {op} to the empty sequence.")]
    ApplyEmptySeq {
        /// Name of the operator applied to the empty sequence (e.g. `Head`).
        op: &'static str,
        /// Source location of the offending application, if known.
        span: Option<Span>,
    },

    /// TLC-compatible evaluation form error (TLC: EC.TLC_MODULE_EVALUATING = 2182)
    /// Format: "Evaluating an expression of the form <form> when s is not a <type>:\n<value>"
    /// Used by Append, Cons, Concat.
    #[error("Evaluating an expression of the form {form} when s is not a {expected_type}:\n{value_display}")]
    EvaluatingError {
        /// Description of the expression form being evaluated (e.g. `Append(s, e)`).
        form: &'static str,
        /// Human-readable description of the type the form required.
        expected_type: &'static str,
        /// Rendered value of the offending operand.
        value_display: String,
        /// Source location of the offending expression, if known.
        span: Option<Span>,
    },

    /// CHOOSE found no witness
    #[error("CHOOSE failed: no value satisfies predicate")]
    ChooseFailed {
        /// Source location of the offending CHOOSE expression, if known.
        span: Option<Span>,
    },

    /// Arity mismatch in operator application
    #[error("Arity mismatch: {op} expects {expected} arguments, got {got}")]
    ArityMismatch {
        /// Name of the operator that was misapplied.
        op: String,
        /// Number of arguments the operator declares.
        expected: usize,
        /// Number of arguments actually supplied.
        got: usize,
        /// Source location of the offending application, if known.
        span: Option<Span>,
    },

    /// Set too large to enumerate
    #[error("Set too large to enumerate (infinite or > limit)")]
    SetTooLarge {
        /// Source location of the offending set expression, if known.
        span: Option<Span>,
    },

    /// TLC-compatible argument type error (TLC: EC.TLC_MODULE_ARGUMENT_ERROR)
    /// Format: "The <position> argument of <op> should be <a/an> <expected>, but instead it is:\n<value>"
    #[error("{}", _format_argument_error(.position, .op, .expected_type, .value_display))]
    ArgumentError {
        /// Ordinal position of the offending argument (e.g. `"first"`, `"second"`).
        position: &'static str,
        /// Name of the operator that rejected the argument.
        op: String,
        /// Human-readable description of the type the operator required.
        expected_type: &'static str,
        /// Rendered value of the rejected argument.
        value_display: String,
        /// Source location of the offending application, if known.
        span: Option<Span>,
    },

    /// Internal evaluation error (bug in evaluator)
    #[error("Internal error: {message}")]
    Internal {
        /// Diagnostic detail describing the internal invariant that was violated.
        message: String,
        /// Source location associated with the failure, if known.
        span: Option<Span>,
    },

    /// Unbounded CHOOSE expression (`CHOOSE x : P(x)` without `\in S`).
    ///
    /// Occurs when CHOOSE has no domain set and thus cannot be evaluated
    /// (requires enumerating all possible values). During compile-time
    /// constant evaluation, this is an expected deferral — the expression
    /// simply cannot be evaluated at compile time.
    /// Part of #2861: replaces `Internal { message: "CHOOSE requires bounded quantification" }`.
    #[error("CHOOSE requires bounded quantification")]
    ChooseUnbounded {
        /// Source location of the unbounded CHOOSE expression, if known.
        span: Option<Span>,
    },

    /// Primed variable evaluated outside next-state context.
    ///
    /// Occurs when a guard expression references a primed variable (e.g., `x'`)
    /// but no next-state environment is bound. In guard checking, this indicates
    /// the expression is an action-level construct, not a pure guard.
    /// Part of #1891: replaces string-matched Internal error.
    #[error("Primed variable cannot be evaluated (no next-state context)")]
    PrimedVariableNotBound {
        /// Source location of the offending primed reference, if known.
        span: Option<Span>,
    },

    /// UNCHANGED evaluated outside next-state context.
    ///
    /// Occurs when UNCHANGED is evaluated without a next-state environment.
    /// In guard checking, this indicates an action-level construct.
    /// Part of #1891: replaces string-matched Internal error.
    #[error("UNCHANGED cannot be evaluated (no next-state context)")]
    UnchangedNotEvaluable {
        /// Source location of the offending UNCHANGED expression, if known.
        span: Option<Span>,
    },

    /// TLC Assert(FALSE, msg) — throws the message string directly.
    /// TLC semantics: the error message IS the second argument, not wrapped.
    /// This enables AssertError(msg, Assert(FALSE, msg)) to return TRUE.
    #[error("{message}")]
    AssertionFailed {
        /// The assertion message; per TLC semantics this is the second argument
        /// to `Assert`, surfaced verbatim rather than wrapped.
        message: String,
        /// Source location of the offending `Assert`, if known.
        span: Option<Span>,
    },

    /// TLCSet("exit", TRUE) was called - early termination requested
    /// Part of #254: Animation and export specs use this for bounded exploration.
    #[error("Model checking terminated by TLCSet(\"exit\", TRUE)")]
    ExitRequested {
        /// Source location of the `TLCSet("exit", TRUE)` call, if known.
        span: Option<Span>,
    },

    /// CASE expression: no arm guard evaluated to TRUE and no OTHER clause.
    /// TLC: Assert.fail("Attempted to evaluate a CASE expression, and none of the conditions were true.")
    /// Part of #1425: Previously silently returned Ok(()), masking spec errors.
    #[error("Attempted to evaluate a CASE expression, and none of the conditions were true.")]
    CaseNoMatch {
        /// Source location of the offending CASE expression, if known.
        span: Option<Span>,
    },

    /// CASE guard evaluation failed.
    ///
    /// This wrapper preserves the underlying error while preventing disabled-action
    /// fallback paths from incorrectly swallowing fatal CASE guard failures.
    /// Display intentionally mirrors the wrapped source error.
    #[error("{source}")]
    CaseGuardError {
        /// The underlying guard-evaluation error, boxed to keep the enum small.
        source: Box<EvalError>,
        /// Source location of the CASE guard that failed, if known.
        span: Option<Span>,
    },
}

impl EvalError {
    // Value-dependent convenience constructors (type_error, one_argument_error,
    // evaluating_error, argument_error) are in value/error_constructors.rs
    // to break the error->value circular dependency (Part of #1269 Phase 2).

    /// Returns the source span attached to this error, if any.
    ///
    /// Every variant carries an optional [`Span`]; this accessor projects it
    /// uniformly so callers can attach location information to diagnostics
    /// without matching on the variant. Returns `None` when the error was
    /// constructed without location information.
    pub fn span(&self) -> Option<Span> {
        match self {
            EvalError::TypeError { span, .. } => *span,
            EvalError::DivisionByZero { span } => *span,
            EvalError::ModulusNotPositive { span, .. } => *span,
            EvalError::UndefinedVar { span, .. } => *span,
            EvalError::UndefinedOp { span, .. } => *span,
            EvalError::NotInDomain { span, .. } => *span,
            EvalError::NoSuchField { span, .. } => *span,
            EvalError::IndexOutOfBounds { span, .. } => *span,
            EvalError::OneArgumentError { span, .. } => *span,
            EvalError::ApplyEmptySeq { span, .. } => *span,
            EvalError::EvaluatingError { span, .. } => *span,
            EvalError::ChooseFailed { span } => *span,
            EvalError::ArityMismatch { span, .. } => *span,
            EvalError::SetTooLarge { span } => *span,
            EvalError::ArgumentError { span, .. } => *span,
            EvalError::Internal { span, .. } => *span,
            EvalError::ChooseUnbounded { span } => *span,
            EvalError::PrimedVariableNotBound { span } => *span,
            EvalError::UnchangedNotEvaluable { span } => *span,
            EvalError::AssertionFailed { span, .. } => *span,
            EvalError::ExitRequested { span } => *span,
            EvalError::CaseNoMatch { span } => *span,
            EvalError::CaseGuardError { span, .. } => *span,
        }
    }
}

/// Convenience alias for results produced while evaluating TLA+ expressions.
pub type EvalResult<T> = Result<T, EvalError>;
