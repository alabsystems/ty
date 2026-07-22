// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! JSON Lines (JSONL) streaming output for model checking events.

use super::types::JsonValue;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// JSONL event types for streaming output
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum JsonlEvent {
    /// Model checking started
    #[serde(rename = "start")]
    Start {
        /// Spec/module name being checked.
        spec: String,
        /// ISO 8601 start timestamp.
        timestamp: String,
    },
    /// Progress update
    #[serde(rename = "progress")]
    Progress {
        /// Distinct states found so far.
        states: usize,
        /// Current BFS depth.
        depth: usize,
        /// Elapsed time in seconds.
        time: f64,
    },
    /// Error detected
    #[serde(rename = "error")]
    Error {
        /// Error type discriminator (e.g. an `error_codes` value).
        error_type: String,
        /// Index of the offending state within the trace, if applicable.
        #[serde(skip_serializing_if = "Option::is_none")]
        state_index: Option<usize>,
    },
    /// State in counterexample
    #[serde(rename = "state")]
    State {
        /// 0-based position of this state in the trace.
        index: usize,
        /// Name of the action that produced this state.
        action: String,
        /// Variable assignment in this state.
        variables: HashMap<String, JsonValue>,
        /// Per-variable `(old, new)` changes from the previous state, if any.
        #[serde(skip_serializing_if = "Option::is_none")]
        diff: Option<HashMap<String, (JsonValue, JsonValue)>>,
    },
    /// Model checking complete
    #[serde(rename = "done")]
    Done {
        /// Final run status string.
        status: String,
        /// Total elapsed time in seconds.
        time: f64,
    },
}

impl JsonlEvent {
    /// Serialize to single JSON line
    pub fn to_jsonl(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}
