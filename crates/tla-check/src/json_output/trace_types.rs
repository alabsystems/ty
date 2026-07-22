// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Counterexample trace and value types for JSON output.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::types::SourceLocation;

/// Counterexample trace
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CounterexampleInfo {
    /// Number of states in the trace.
    pub length: usize,
    /// The states, ordered from the initial state to the violating state.
    pub states: Vec<StateInfo>,
    /// For liveness violations: index where the cycle begins
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loop_start: Option<usize>,
}

/// A single state in a trace
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StateInfo {
    /// 0-based position of this state within the trace.
    pub index: usize,
    /// Hex fingerprint of the state, if recorded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<String>,
    /// The action that produced this state from its predecessor.
    pub action: ActionRef,
    /// Full variable assignment in this state, keyed by variable name.
    pub variables: HashMap<String, JsonValue>,
    /// Diff against the previous state, populated for non-initial states.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diff_from_previous: Option<StateDiff>,
}

/// Reference to an action
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ActionRef {
    /// Name of the action (or the synthetic initial-state label).
    pub name: String,
    /// Type: "initial", "next"
    #[serde(rename = "type")]
    pub action_type: String,
    /// Source location of the action definition, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<SourceLocation>,
}

/// Diff between two states
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StateDiff {
    /// Variables whose value changed, keyed by name, with old and new values.
    pub changed: HashMap<String, ValueChange>,
    /// Names of variables whose value was unchanged.
    pub unchanged: Vec<String>,
}

/// A value change
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ValueChange {
    /// Value in the predecessor state.
    pub from: JsonValue,
    /// Value in the successor state.
    pub to: JsonValue,
}

/// Typed JSON value representation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", content = "value")]
#[non_exhaustive]
pub enum JsonValue {
    /// A TLA+ boolean.
    #[serde(rename = "bool")]
    Bool(bool),
    /// An integer that fits in an `i64`.
    #[serde(rename = "int")]
    Int(i64),
    /// An integer too large for `i64`, encoded as a decimal string.
    #[serde(rename = "big_int")]
    BigInt(String),
    /// A TLA+ string literal.
    #[serde(rename = "string")]
    String(String),
    /// A set, rendered as an unordered list of elements.
    #[serde(rename = "set")]
    Set(Vec<JsonValue>),
    /// A sequence (1-indexed tuple of elements).
    #[serde(rename = "seq")]
    Seq(Vec<JsonValue>),
    /// A record, keyed by field name.
    #[serde(rename = "record")]
    Record(HashMap<String, JsonValue>),
    /// A function value, given by its domain and explicit domain→range mapping.
    #[serde(rename = "function")]
    Function {
        /// The function's domain elements.
        domain: Vec<JsonValue>,
        /// Domain-to-range pairs covering every domain element.
        mapping: Vec<(JsonValue, JsonValue)>,
    },
    /// A fixed-arity tuple.
    #[serde(rename = "tuple")]
    Tuple(Vec<JsonValue>),
    /// An uninterpreted model value (e.g. a CONSTANT symbol).
    #[serde(rename = "model_value")]
    ModelValue(String),
    /// An integer interval `lo..hi`, with bounds as decimal strings.
    #[serde(rename = "interval")]
    Interval {
        /// Inclusive lower bound, as a decimal string.
        lo: String,
        /// Inclusive upper bound, as a decimal string.
        hi: String,
    },
    /// An undefined/error value.
    #[serde(rename = "undefined")]
    Undefined,
}
