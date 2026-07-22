// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Error codes and diagnostic message construction for JSON output.

use super::types::DiagnosticMessage;

/// Stable string codes for the `code`/`error_code` fields of structured JSON output.
///
/// Codes use a prefix convention: `TLC_` model-checker results, `CFG_` config
/// parsing, `TLA_` source parsing, `SYS_` runtime/system, `TRACE_` trace
/// validation. Centralizing the literals here keeps emitters and parsers in sync.
pub mod error_codes {
    // Model checker errors (TLC_*)
    /// A deadlock was reached (a state with no enabled successors).
    pub const TLC_DEADLOCK: &str = "TLC_DEADLOCK";
    /// An invariant was violated.
    pub const TLC_INVARIANT_VIOLATED: &str = "TLC_INVARIANT_VIOLATED";
    /// A temporal PROPERTY was violated.
    pub const TLC_PROPERTY_VIOLATED: &str = "TLC_PROPERTY_VIOLATED";
    /// A liveness property was violated.
    pub const TLC_LIVENESS_VIOLATED: &str = "TLC_LIVENESS_VIOLATED";
    /// The liveness checker cannot handle the given temporal formula.
    pub const TLC_LIVE_CANNOT_HANDLE_FORMULA: &str = "TLC_LIVE_CANNOT_HANDLE_FORMULA";
    /// The liveness formula is a tautology (trivially true, nothing to check).
    pub const TLC_LIVE_FORMULA_TAUTOLOGY: &str = "TLC_LIVE_FORMULA_TAUTOLOGY";
    /// Expression evaluation failed during checking.
    pub const TLC_EVAL_ERROR: &str = "TLC_EVAL_ERROR";
    /// A type mismatch was detected during evaluation.
    pub const TLC_TYPE_MISMATCH: &str = "TLC_TYPE_MISMATCH";
    /// Reference to an undefined variable.
    pub const TLC_UNDEFINED_VAR: &str = "TLC_UNDEFINED_VAR";
    /// Reference to an undefined operator.
    pub const TLC_UNDEFINED_OP: &str = "TLC_UNDEFINED_OP";
    /// A configured exploration limit (states/depth/time) was reached.
    pub const TLC_LIMIT_REACHED: &str = "TLC_LIMIT_REACHED";
    /// Guard-evaluation errors were encountered and suppressed.
    pub const TLC_GUARD_ERRORS_SUPPRESSED: &str = "TLC_GUARD_ERRORS_SUPPRESSED";
    /// Vacuity gate: the run proved nothing (empty/never-exercised/unsat basis).
    /// Maps to the `VACUOUS` verdict (exit code 3). Design: TRUST_VACUITY_GATE.
    pub const TLC_VACUOUS: &str = "TLC_VACUOUS";

    // Configuration errors (CFG_*)
    /// The `.cfg` file could not be parsed.
    pub const CFG_PARSE_ERROR: &str = "CFG_PARSE_ERROR";
    /// The config did not specify an `INIT` predicate.
    pub const CFG_MISSING_INIT: &str = "CFG_MISSING_INIT";
    /// The config did not specify a `NEXT` action.
    pub const CFG_MISSING_NEXT: &str = "CFG_MISSING_NEXT";
    /// The config used syntax that is not yet supported.
    pub const CFG_UNSUPPORTED_SYNTAX: &str = "CFG_UNSUPPORTED_SYNTAX";

    // TLA+ parsing errors (TLA_*)
    /// The TLA+ source could not be parsed.
    pub const TLA_PARSE_ERROR: &str = "TLA_PARSE_ERROR";
    /// The parsed TLA+ source could not be lowered to the internal IR.
    pub const TLA_LOWER_ERROR: &str = "TLA_LOWER_ERROR";

    // System errors (SYS_*)
    /// A liveness-checking subsystem error occurred.
    pub const SYS_LIVENESS_ERROR: &str = "SYS_LIVENESS_ERROR";
    /// The liveness checker failed at runtime.
    pub const SYS_LIVENESS_RUNTIME_FAILURE: &str = "SYS_LIVENESS_RUNTIME_FAILURE";
    /// An I/O error occurred.
    pub const SYS_IO_ERROR: &str = "SYS_IO_ERROR";
    /// The run timed out.
    pub const SYS_TIMEOUT: &str = "SYS_TIMEOUT";
    /// Checker setup failed before exploration began.
    pub const SYS_SETUP_ERROR: &str = "SYS_SETUP_ERROR";
    /// The result was withheld because the engine path did not meet the soundness gate.
    pub const SYS_SOUNDNESS_GATED: &str = "SYS_SOUNDNESS_GATED";
    /// The result was withheld because the engine path did not meet the completeness gate.
    pub const SYS_COMPLETENESS_GATED: &str = "SYS_COMPLETENESS_GATED";

    // Backend / capability statuses — these are NOT error codes, they are
    // values of the `status` / `error_type` fields in the JSON output. They
    // live in this module so every emitter and every parser routes through
    // the same string. See docs/mcc-2026/qualification-1/analysis.md for
    // why duplicated literals are dangerous.
    /// `status` value indicating the selected backend was unavailable.
    pub const STATUS_BACKEND_UNAVAILABLE: &str = "backend_unavailable";
    /// `error_type` value indicating an invariant violation.
    pub const ERROR_TYPE_INVARIANT_VIOLATION: &str = "invariant_violation";

    // Trace validation errors (TRACE_*)
    //
    // These are intended for semantic trace validation / diagnostics, not model checking results.
    /// No binder was found for a trace action-label parameter.
    pub const TRACE_PARAM_BIND_MISSING_BINDER: &str = "TRACE_PARAM_BIND_MISSING_BINDER";
    /// Multiple binders matched a trace action-label parameter.
    pub const TRACE_PARAM_BIND_AMBIGUOUS_BINDER: &str = "TRACE_PARAM_BIND_AMBIGUOUS_BINDER";
    /// No call site was found for a trace action-label parameter.
    pub const TRACE_PARAM_BIND_MISSING_CALLSITE: &str = "TRACE_PARAM_BIND_MISSING_CALLSITE";
    /// Multiple call sites matched a trace action-label parameter.
    pub const TRACE_PARAM_BIND_AMBIGUOUS_CALLSITE: &str = "TRACE_PARAM_BIND_AMBIGUOUS_CALLSITE";
    /// A call-site argument form is not supported for parameter binding.
    pub const TRACE_PARAM_BIND_UNSUPPORTED_CALLSITE_ARG: &str =
        "TRACE_PARAM_BIND_UNSUPPORTED_CALLSITE_ARG";
    /// A binder pattern is not supported for parameter binding.
    pub const TRACE_PARAM_BIND_UNSUPPORTED_BINDER_PATTERN: &str =
        "TRACE_PARAM_BIND_UNSUPPORTED_BINDER_PATTERN";
}

impl DiagnosticMessage {
    /// Convert a rewrite-backend action-label param binding error into a structured diagnostic.
    pub fn from_action_label_param_bind_error(err: &crate::ActionLabelParamBindError) -> Self {
        fn span_payload(span: tla_core::Span) -> serde_json::Value {
            serde_json::json!({
                "file_id": span.file.0,
                "start": span.start,
                "end": span.end,
            })
        }

        fn binder_site_payload(site: &crate::BoundVarSite) -> serde_json::Value {
            serde_json::json!({
                "name": site.name,
                "span": span_payload(site.span),
            })
        }

        use crate::ActionLabelParamBindErrorKind;

        let outer_binders: Vec<serde_json::Value> =
            err.outer_binders.iter().map(binder_site_payload).collect();

        let (code, kind, param) = match &err.kind {
            ActionLabelParamBindErrorKind::MissingBinder { param } => (
                error_codes::TRACE_PARAM_BIND_MISSING_BINDER,
                "missing_binder",
                param.as_str(),
            ),
            ActionLabelParamBindErrorKind::AmbiguousBinder { param, .. } => (
                error_codes::TRACE_PARAM_BIND_AMBIGUOUS_BINDER,
                "ambiguous_binder",
                param.as_str(),
            ),
            ActionLabelParamBindErrorKind::MissingCallsite { param } => (
                error_codes::TRACE_PARAM_BIND_MISSING_CALLSITE,
                "missing_callsite",
                param.as_str(),
            ),
            ActionLabelParamBindErrorKind::AmbiguousCallsite { param, .. } => (
                error_codes::TRACE_PARAM_BIND_AMBIGUOUS_CALLSITE,
                "ambiguous_callsite",
                param.as_str(),
            ),
            ActionLabelParamBindErrorKind::UnsupportedCallsiteArg { param, .. } => (
                error_codes::TRACE_PARAM_BIND_UNSUPPORTED_CALLSITE_ARG,
                "unsupported_callsite_arg",
                param.as_str(),
            ),
        };

        let mut payload = serde_json::Map::new();
        payload.insert("label".to_string(), serde_json::json!(err.label));
        payload.insert(
            "operator_raw".to_string(),
            serde_json::json!(err.operator_raw),
        );
        payload.insert(
            "action_name".to_string(),
            serde_json::json!(err.action_name),
        );
        payload.insert(
            "action_id".to_string(),
            serde_json::json!(err.action_id.to_string()),
        );
        payload.insert(
            "action_span".to_string(),
            serde_json::json!(span_payload(err.action_span)),
        );
        payload.insert(
            "outer_binders".to_string(),
            serde_json::Value::Array(outer_binders),
        );
        payload.insert("kind".to_string(), serde_json::json!(kind));
        payload.insert("param".to_string(), serde_json::json!(param));

        match &err.kind {
            ActionLabelParamBindErrorKind::AmbiguousBinder { matches, .. } => {
                let matches_payload: Vec<serde_json::Value> =
                    matches.iter().map(binder_site_payload).collect();
                payload.insert(
                    "matches".to_string(),
                    serde_json::Value::Array(matches_payload),
                );
            }
            ActionLabelParamBindErrorKind::AmbiguousCallsite { callsites, .. } => {
                let spans_payload: Vec<serde_json::Value> =
                    callsites.iter().copied().map(span_payload).collect();
                payload.insert(
                    "callsites".to_string(),
                    serde_json::Value::Array(spans_payload),
                );
            }
            ActionLabelParamBindErrorKind::UnsupportedCallsiteArg {
                position,
                arg,
                arg_span,
                ..
            } => {
                payload.insert("position".to_string(), serde_json::json!(position));
                payload.insert("arg".to_string(), serde_json::json!(arg));
                payload.insert(
                    "arg_span".to_string(),
                    serde_json::json!(span_payload(*arg_span)),
                );
            }
            _ => {}
        }

        Self {
            code: code.to_string(),
            message: err.to_string(),
            location: None,
            suggestion: Some(err.suggestion_text().to_string()),
            payload: Some(serde_json::Value::Object(payload)),
        }
    }
}
