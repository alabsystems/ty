// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Trace validation input parsing (JSON + JSONL).
//!
//! This is an IO layer that incrementally emits a header and ordered steps to a sink, without
//! coupling to the trace validation engine itself.

mod parse;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::json_output::JsonValue;

pub use parse::{read_trace_events, read_trace_json, read_trace_jsonl};

/// A caller's requested trace input format, before resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceInputFormatSelection {
    /// Automatically select a format.
    ///
    /// Resolution uses a best-effort heuristic based on a `TraceSourceHint`:
    /// - `.jsonl` extension => JSONL
    /// - otherwise => JSON
    ///
    /// Use `resolve_trace_input_format()` to map this into a resolved `TraceInputFormat`.
    Auto,
    /// JSON object form `{ version, module, variables, steps: [...] }`.
    ///
    /// Note: this currently deserializes the full `steps` array into memory.
    Json,
    /// JSON Lines form: one event object per line (`header` then `step`).
    Jsonl,
}

/// A concrete trace input format, after [`resolve_trace_input_format`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceInputFormat {
    /// JSON object form `{ version, module, variables, steps: [...] }`.
    ///
    /// Note: this currently deserializes the full `steps` array into memory.
    Json,
    /// JSON Lines form: one event object per line (`header` then `step`).
    Jsonl,
}

/// A hint about where trace input is coming from, used for format auto-detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceSourceHint<'a> {
    /// Input is a file at the given path (extension drives auto-detection).
    Path(&'a Path),
    /// Input is standard input.
    Stdin,
    /// Source is unknown.
    Unknown,
}

/// Resolves a requested [`TraceInputFormatSelection`] into a concrete
/// [`TraceInputFormat`], using `hint` to break ties for `Auto`.
///
/// `Auto` with a `.jsonl` path resolves to JSONL; every other case resolves to JSON.
pub fn resolve_trace_input_format(
    selection: TraceInputFormatSelection,
    hint: TraceSourceHint<'_>,
) -> TraceInputFormat {
    match selection {
        TraceInputFormatSelection::Json => TraceInputFormat::Json,
        TraceInputFormatSelection::Jsonl => TraceInputFormat::Jsonl,
        TraceInputFormatSelection::Auto => match hint {
            TraceSourceHint::Path(p) if is_jsonl_extension(p) => TraceInputFormat::Jsonl,
            TraceSourceHint::Path(_) | TraceSourceHint::Stdin | TraceSourceHint::Unknown => {
                TraceInputFormat::Json
            }
        },
    }
}

/// Header of a trace: format version, module name, and declared variables.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TraceHeader {
    /// Trace format version string.
    pub version: String,
    /// Name of the spec module the trace was produced for.
    pub module: String,
    /// Names of the state variables present in each step.
    pub variables: Vec<String>,
}

/// An action label attached to a trace step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceActionLabel {
    /// The action label name.
    pub name: String,
    /// Optional encoded parameter values for the action.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

/// A single state in a trace, optionally annotated with the action that reached it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStep {
    /// 0-based step index, if present (validated against position).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    /// Variable assignment for this state, keyed by variable name.
    pub state: HashMap<String, JsonValue>,
    /// Action label for the transition into this state (absent on step 0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<TraceActionLabel>,
}

/// Sink that receives the header and each step as a trace is parsed.
///
/// Implementations let parsing stream steps without buffering the whole trace.
pub trait TraceEventSink {
    /// Called once with the trace header before any steps.
    fn on_header(&mut self, header: TraceHeader);
    /// Called once per step, in order.
    fn on_step(&mut self, step: TraceStep);
}

/// Error parsing or validating trace input (JSON or JSONL).
#[derive(Debug, thiserror::Error)]
pub enum TraceParseError {
    /// An I/O error occurred while reading the input.
    #[error("trace input IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON decoding failed without positional context.
    #[error("trace JSON decode failed: {0}")]
    JsonDecode(#[from] serde_json::Error),

    /// JSON decoding failed, with file and line/column context.
    #[error("trace JSON decode failed at {path} (line {line}, column {column}): {source}")]
    JsonDecodePath {
        /// Input path.
        path: String,
        /// 1-based line of the error.
        line: usize,
        /// 1-based column of the error.
        column: usize,
        /// Underlying decode error.
        #[source]
        source: serde_json::Error,
    },

    /// A JSONL line failed to decode.
    #[error(
        "trace JSONL decode failed at {path} (line {line_no}, column {column}): {source} (line prefix: {raw_line_prefix})"
    )]
    JsonlDecode {
        /// 1-based line number of the failing line.
        line_no: usize,
        /// Input path.
        path: String,
        /// 1-based column of the error.
        column: usize,
        /// Underlying decode error.
        #[source]
        source: serde_json::Error,
        /// Truncated prefix of the offending raw line, for diagnostics.
        raw_line_prefix: String,
    },

    /// A JSONL event carried an unrecognized `type`.
    #[error("trace JSONL encountered unknown event type {ty:?} (line {line_no})")]
    JsonlUnknownEventType {
        /// 1-based line number.
        line_no: usize,
        /// The unrecognized event type string.
        ty: String,
    },

    /// A JSONL step appeared before the header.
    #[error("trace JSONL missing header before first step (line {line_no})")]
    JsonlMissingHeader {
        /// 1-based line number of the premature step.
        line_no: usize,
    },

    /// JSONL input ended without ever providing a header.
    #[error("trace JSONL missing header (no events found)")]
    JsonlMissingHeaderAtEof,

    /// A JSONL header appeared after one was already seen.
    #[error("trace JSONL encountered a second header (line {line_no})")]
    JsonlUnexpectedHeader {
        /// 1-based line number of the duplicate header.
        line_no: usize,
    },

    /// Internal invariant violation: a buffered header was unexpectedly absent.
    #[error("trace JSONL internal error: buffered header unexpectedly missing before step (line {line_no})")]
    JsonlMissingBufferedHeader {
        /// 1-based line number.
        line_no: usize,
    },

    /// The trace contained no steps (expected at least step 0).
    #[error("trace input missing any steps (expected step 0) ({where_})")]
    MissingAnySteps {
        /// Context describing where the error was detected.
        where_: String,
    },

    /// A step's declared index did not match its position.
    #[error("trace step index mismatch at {where_}: expected {expected}, got {got}")]
    StepIndexMismatch {
        /// Context describing where the error was detected.
        where_: String,
        /// Expected index (position).
        expected: usize,
        /// Index the step declared.
        got: usize,
    },

    /// A step referenced a variable not declared in the header.
    #[error("trace step references unknown variable {var:?} at {where_}")]
    UnknownVariable {
        /// Context describing where the error was detected.
        where_: String,
        /// The undeclared variable name.
        var: String,
    },

    /// Step 0 carried an action label, which is not allowed.
    #[error("trace step 0 must not have an action label ({where_})")]
    ActionOnInitialStep {
        /// Context describing where the error was detected.
        where_: String,
    },

    /// The header declared the same variable twice.
    #[error("trace header contains duplicate variable {var:?} ({where_})")]
    DuplicateVariable {
        /// Context describing where the error was detected.
        where_: String,
        /// The duplicated variable name.
        var: String,
    },
}

fn is_jsonl_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("jsonl"))
}
