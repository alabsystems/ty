// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared JSON codec for converting between `Value` and `JsonValue`.
//!
//! This exists to support both JSON output (`JsonOutput`) and trace-input parsing / validation.

mod decode;

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use num_bigint::BigInt;
use num_traits::ToPrimitive;

use crate::json_output::JsonValue;
use crate::value::{IntIntervalFunc, Value};

pub use decode::{json_to_value, json_to_value_with_path};

/// Error decoding a [`JsonValue`] into a TLA+ [`Value`].
#[derive(Debug, thiserror::Error)]
pub enum JsonValueDecodeError {
    /// A `big_int` string was not a valid decimal integer.
    #[error("invalid decimal BigInt literal: {value:?}")]
    InvalidBigInt {
        /// The offending string.
        value: String,
    },

    /// Encountered a [`JsonValue::Undefined`], which has no TLA+ value.
    #[error("cannot decode undefined JSON value")]
    Undefined,

    /// A model value could not be constructed (e.g. interner failure).
    #[error("failed to construct model value: {0}")]
    ModelValueError(String),

    /// A function mapping listed the same domain key twice.
    #[error("function mapping contains duplicate key: {key:?}")]
    DuplicateFunctionKey {
        /// The duplicated key.
        key: String,
    },

    /// A function domain listed the same key twice.
    #[error("function domain contains duplicate key: {key:?}")]
    DuplicateFunctionDomainKey {
        /// The duplicated key.
        key: String,
    },

    /// A function domain key has no corresponding mapping entry.
    #[error("function domain contains a key with no mapping: {key:?}")]
    FunctionDomainMissingKey {
        /// The unmapped domain key.
        key: String,
    },

    /// A function mapping references a key absent from the declared domain.
    #[error("function mapping contains a key not present in domain: {key:?}")]
    FunctionMappingKeyNotInDomain {
        /// The out-of-domain key.
        key: String,
    },
}

/// A [`JsonValueDecodeError`] annotated with the JSON path where it occurred.
#[derive(Debug, thiserror::Error)]
#[error("decode failed at {path}: {source}")]
pub struct JsonValueDecodeErrorAtPath {
    /// JSON path (e.g. `steps[2].state.x`) of the failing value.
    pub path: String,
    /// The underlying decode error.
    #[source]
    pub source: JsonValueDecodeError,
}

/// Error type for converting trace `params` (generic JSON) to typed TLA values.
#[derive(Debug, thiserror::Error)]
pub enum ParamsDecodeError {
    /// The `params` value was not a JSON object.
    #[error("action params must be a JSON object, got {actual_type}")]
    NotAnObject {
        /// The actual JSON type encountered.
        actual_type: &'static str,
    },
    /// A param value was not a valid [`JsonValue`] encoding.
    #[error("invalid JsonValue encoding for param {key:?}: {source}")]
    InvalidJsonValue {
        /// The param key whose value failed to decode.
        key: String,
        /// Underlying serde error.
        #[source]
        source: serde_json::Error,
    },
}

/// Returns a human-readable type name for a serde_json::Value.
fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Convert trace params (serde_json::Value) to typed params (HashMap<String, JsonValue>).
///
/// Expects `params` to be a JSON object where each value is a valid JsonValue encoding.
/// Returns empty HashMap for JSON null (treated as "no params").
pub fn params_from_json(
    params: &serde_json::Value,
) -> Result<HashMap<String, JsonValue>, ParamsDecodeError> {
    // JSON null = no params (per edge case semantics)
    if params.is_null() {
        return Ok(HashMap::new());
    }
    let obj = params
        .as_object()
        .ok_or_else(|| ParamsDecodeError::NotAnObject {
            actual_type: json_type_name(params),
        })?;
    let mut result = HashMap::with_capacity(obj.len());
    for (key, val) in obj {
        let typed: JsonValue = serde_json::from_value(val.clone()).map_err(|e| {
            ParamsDecodeError::InvalidJsonValue {
                key: key.clone(),
                source: e,
            }
        })?;
        result.insert(key.clone(), typed);
    }
    Ok(result)
}

/// Converts a TLA+ [`Value`] into its [`JsonValue`] representation.
///
/// Lazy or potentially-infinite set/function values (e.g. `SUBSET S`, `[A -> B]`,
/// `STRING`) that cannot be enumerated cheaply are rendered as [`JsonValue::Undefined`].
pub fn value_to_json(value: &Value) -> JsonValue {
    match value {
        Value::Bool(b) => JsonValue::Bool(*b),
        Value::SmallInt(n) => JsonValue::Int(*n),
        Value::Int(n) => match n.to_i64() {
            Some(small) => JsonValue::Int(small),
            None => JsonValue::BigInt(n.to_string()),
        },
        Value::String(s) => JsonValue::String(s.to_string()),
        Value::Set(elements) => JsonValue::Set(elements.iter().map(value_to_json).collect()),
        Value::Tuple(elements) => JsonValue::Tuple(elements.iter().map(value_to_json).collect()),
        Value::Seq(elements) => JsonValue::Seq(elements.iter().map(value_to_json).collect()),
        Value::Record(fields) => JsonValue::Record(
            fields
                .iter_str()
                .map(|(k, v)| (k.to_string(), value_to_json(v)))
                .collect(),
        ),
        Value::Func(func) => JsonValue::Function {
            domain: func.domain_iter().map(value_to_json).collect(),
            mapping: func
                .mapping_iter()
                .map(|(k, v)| (value_to_json(k), value_to_json(v)))
                .collect(),
        },
        // Bag serializes exactly like the equivalent Func.
        Value::Bag(bag) => JsonValue::Function {
            domain: bag.elems().iter().map(value_to_json).collect(),
            mapping: bag
                .entries()
                .map(|(k, c)| (value_to_json(k), JsonValue::Int(c)))
                .collect(),
        },
        Value::IntFunc(func) => JsonValue::Function {
            domain: (IntIntervalFunc::min(func)..=IntIntervalFunc::max(func))
                .map(JsonValue::Int)
                .collect(),
            mapping: (IntIntervalFunc::min(func)..=IntIntervalFunc::max(func))
                .zip(func.values().iter())
                .map(|(k, v)| (JsonValue::Int(k), value_to_json(v)))
                .collect(),
        },
        Value::ModelValue(id) => JsonValue::ModelValue(id.to_string()),
        Value::Interval(iv) => JsonValue::Interval {
            lo: iv.low().to_string(),
            hi: iv.high().to_string(),
        },
        // For complex lazy sets that might be expensive/infinite, represent as undefined.
        Value::FuncSet(_)
        | Value::Subset(_)
        | Value::RecordSet(_)
        | Value::TupleSet(_)
        | Value::SetCup(_)
        | Value::SetCap(_)
        | Value::SetDiff(_)
        | Value::SetPred(_)
        | Value::KSubset(_)
        | Value::BigUnion(_)
        | Value::LazyFunc(_)
        | Value::Closure(_)
        | Value::StringSet
        | Value::AnySet
        | Value::SeqSet(_) => JsonValue::Undefined,
        _ => JsonValue::Undefined,
    }
}

pub(crate) fn parse_big_int_decimal(s: &str) -> Result<BigInt, JsonValueDecodeError> {
    s.parse::<BigInt>()
        .map_err(|_| JsonValueDecodeError::InvalidBigInt {
            value: s.to_string(),
        })
}

pub(crate) fn json_key_string(key: &JsonValue) -> String {
    match key {
        JsonValue::Bool(b) => b.to_string(),
        JsonValue::Int(n) => n.to_string(),
        JsonValue::BigInt(s) => s.clone(),
        JsonValue::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| format!("{other:?}")),
    }
}
