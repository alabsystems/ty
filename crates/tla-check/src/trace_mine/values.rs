// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Rendering observed trace values as TLA+ literals.

use std::collections::BTreeSet;

use crate::json_output::JsonValue;

use super::MineError;

/// Rendering side effects accumulated while emitting values: which stdlib
/// modules the module must EXTEND and which model-value constants it needs.
#[derive(Debug, Default, Clone)]
pub struct RenderCtx {
    /// `EXTENDS Integers` required (integer literals / arithmetic / ranges).
    pub needs_integers: bool,
    /// `EXTENDS TLC` required (`:>` / `@@` function literals).
    pub needs_tlc: bool,
    /// Model values observed in the corpus (become `CONSTANTS`).
    pub model_values: BTreeSet<String>,
}

/// Render an observed value as a TLA+ literal.
///
/// # Errors
///
/// [`MineError::UnsupportedValue`] for values with no literal rendering
/// (`undefined`, empty records, model values that are not identifiers).
pub(crate) fn render_value(
    value: &JsonValue,
    var: &str,
    ctx: &mut RenderCtx,
) -> Result<String, MineError> {
    let unsupported = |why: &str| MineError::UnsupportedValue {
        var: var.to_string(),
        why: why.to_string(),
    };
    Ok(match value {
        JsonValue::Bool(b) => if *b { "TRUE" } else { "FALSE" }.to_string(),
        JsonValue::Int(i) => {
            ctx.needs_integers = true;
            i.to_string()
        }
        JsonValue::BigInt(s) => {
            ctx.needs_integers = true;
            s.clone()
        }
        JsonValue::String(s) => format!("\"{}\"", escape_tla_string(s)),
        JsonValue::Set(elems) => {
            let mut rendered = elems
                .iter()
                .map(|e| render_value(e, var, ctx))
                .collect::<Result<Vec<_>, _>>()?;
            rendered.sort();
            rendered.dedup();
            format!("{{{}}}", rendered.join(", "))
        }
        JsonValue::Seq(elems) | JsonValue::Tuple(elems) => {
            let rendered = elems
                .iter()
                .map(|e| render_value(e, var, ctx))
                .collect::<Result<Vec<_>, _>>()?;
            format!("<<{}>>", rendered.join(", "))
        }
        JsonValue::Record(fields) => {
            if fields.is_empty() {
                return Err(unsupported("empty record has no TLA+ literal"));
            }
            let mut keys: Vec<&String> = fields.keys().collect();
            keys.sort();
            let rendered = keys
                .iter()
                .map(|k| Ok(format!("{k} |-> {}", render_value(&fields[*k], var, ctx)?)))
                .collect::<Result<Vec<_>, MineError>>()?;
            format!("[{}]", rendered.join(", "))
        }
        JsonValue::Function { mapping, .. } => {
            if mapping.is_empty() {
                // The canonical empty function.
                "[e \\in {} |-> e]".to_string()
            } else {
                ctx.needs_tlc = true;
                let mut pairs = mapping
                    .iter()
                    .map(|(d, r)| {
                        Ok(format!(
                            "({} :> {})",
                            render_value(d, var, ctx)?,
                            render_value(r, var, ctx)?
                        ))
                    })
                    .collect::<Result<Vec<_>, MineError>>()?;
                pairs.sort();
                pairs.join(" @@ ")
            }
        }
        JsonValue::ModelValue(name) => {
            if !is_tla_identifier(name) {
                return Err(unsupported(&format!(
                    "model value {name:?} is not a TLA+ identifier"
                )));
            }
            ctx.model_values.insert(name.clone());
            name.clone()
        }
        JsonValue::Interval { lo, hi } => {
            ctx.needs_integers = true;
            format!("{lo}..{hi}")
        }
        JsonValue::Undefined => return Err(unsupported("undefined value")),
    })
}

/// A deterministic equality key for an observed value.
///
/// Two values compare equal iff their keys match. Uses the debug encoding
/// (not the TLA+ rendering) so keying never fails on unsupported values.
pub(crate) fn value_key(value: &JsonValue) -> String {
    match value {
        JsonValue::Set(elems) => {
            let mut keys: Vec<String> = elems.iter().map(value_key).collect();
            keys.sort();
            keys.dedup();
            format!("set{{{}}}", keys.join(","))
        }
        JsonValue::Record(fields) => {
            let mut keys: Vec<&String> = fields.keys().collect();
            keys.sort();
            let body: Vec<String> = keys
                .iter()
                .map(|k| format!("{k}:{}", value_key(&fields[*k])))
                .collect();
            format!("rec[{}]", body.join(","))
        }
        JsonValue::Function { mapping, .. } => {
            let mut pairs: Vec<String> = mapping
                .iter()
                .map(|(d, r)| format!("{}->{}", value_key(d), value_key(r)))
                .collect();
            pairs.sort();
            format!("fun({})", pairs.join(","))
        }
        JsonValue::Seq(elems) => {
            let keys: Vec<String> = elems.iter().map(value_key).collect();
            format!("seq<{}>", keys.join(","))
        }
        JsonValue::Tuple(elems) => {
            let keys: Vec<String> = elems.iter().map(value_key).collect();
            format!("tup<{}>", keys.join(","))
        }
        other => format!("{other:?}"),
    }
}

/// Extract an `i64` when the value is an in-range integer.
pub(crate) fn as_int(value: &JsonValue) -> Option<i64> {
    match value {
        JsonValue::Int(i) => Some(*i),
        _ => None,
    }
}

/// Whether `s` can be used verbatim as a TLA+ identifier.
pub(crate) fn is_tla_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    let starts_ok = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
    starts_ok
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !is_tla_reserved(s)
        && s.chars().any(|c| c.is_ascii_alphabetic())
}

/// Sanitize an arbitrary action-label into a TLA+ identifier.
pub(crate) fn sanitize_identifier(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if !out
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
    {
        out.insert(0, 'A');
        out.insert(1, '_');
    }
    if !is_tla_identifier(&out) {
        out = format!("A_{out}");
    }
    out
}

/// TLA+ reserved words that cannot serve as identifiers.
fn is_tla_reserved(s: &str) -> bool {
    matches!(
        s,
        "ASSUME"
            | "ASSUMPTION"
            | "AXIOM"
            | "BOOLEAN"
            | "CASE"
            | "CHOOSE"
            | "CONSTANT"
            | "CONSTANTS"
            | "DOMAIN"
            | "ELSE"
            | "ENABLED"
            | "EXCEPT"
            | "EXTENDS"
            | "FALSE"
            | "IF"
            | "IN"
            | "INSTANCE"
            | "LAMBDA"
            | "LET"
            | "LOCAL"
            | "MODULE"
            | "OTHER"
            | "SF_"
            | "STRING"
            | "SUBSET"
            | "THEN"
            | "THEOREM"
            | "TRUE"
            | "UNCHANGED"
            | "UNION"
            | "VARIABLE"
            | "VARIABLES"
            | "WF_"
            | "WITH"
    )
}

/// Escape a string for a TLA+ string literal.
fn escape_tla_string(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}
