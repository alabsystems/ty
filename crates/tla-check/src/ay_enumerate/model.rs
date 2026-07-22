// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#[cfg(feature = "ay")]
use std::collections::HashMap;
#[cfg(feature = "ay")]
use std::sync::Arc;

#[cfg(feature = "ay")]
use num_traits::ToPrimitive;

#[cfg(feature = "ay")]
use super::{AYEnumError, AYEnumResult, VarInfo, VarSort};
#[cfg(feature = "ay")]
use crate::state::State;
#[cfg(feature = "ay")]
use crate::value::intern_string;
#[cfg(feature = "ay")]
use crate::Value;

/// Read a scalar (Bool, Int, or String) value from a ay model.
/// Returns an error for compound types (Function, Tuple, Heterogeneous).
#[cfg(feature = "ay")]
fn model_get_scalar(
    model: &tla_ay::Model,
    var_name: &str,
    sort: &VarSort,
    string_reverse_map: &HashMap<i64, String>,
) -> AYEnumResult<Value> {
    match sort {
        VarSort::Bool => {
            let b = model.bool_val(var_name).ok_or_else(|| {
                AYEnumError::InvalidModel(format!("missing bool value for {}", var_name))
            })?;
            Ok(Value::Bool(b))
        }
        VarSort::Int => {
            let n = model.int_val(var_name).ok_or_else(|| {
                AYEnumError::InvalidModel(format!("missing int value for {}", var_name))
            })?;
            // SOUNDNESS (#633/#634): integers that fit in i64 MUST be stored as
            // Value::SmallInt, not Value::Int. The two encode to the same TLC
            // fingerprint but different canonical state-slot payloads; emitting a
            // non-normalized Value::Int here makes ay-enumerated initial states
            // collide (same fingerprint, mismatched canonical payload) with the
            // SmallInt successors produced by the explicit evaluator, tripping the
            // fail-closed prepared_fingerprint_admission layer. Route through the
            // normalizing big_int() constructor to preserve the Value invariant.
            Ok(Value::big_int(n.clone()))
        }
        VarSort::String { .. } => {
            let id = model.int_val(var_name).ok_or_else(|| {
                AYEnumError::InvalidModel(format!("missing string value for {}", var_name))
            })?;
            let id = id.to_i64().ok_or_else(|| {
                AYEnumError::InvalidModel(format!(
                    "string ID for {} does not fit in i64: {}",
                    var_name, id
                ))
            })?;
            let s = string_reverse_map.get(&id).ok_or_else(|| {
                AYEnumError::InvalidModel(format!("unknown string ID {} for {}", id, var_name))
            })?;
            // Part of #3287: Route through intern_string() for eager TLC
            // token assignment, matching TLC's UniqueString.uniqueStringOf().
            Ok(Value::String(intern_string(s.as_str())))
        }
        VarSort::Function { .. } | VarSort::Tuple { .. } | VarSort::Heterogeneous { .. } => {
            Err(AYEnumError::UnsupportedVarType {
                var: var_name.to_string(),
                reason: format!("compound type {:?} not supported as scalar", sort),
            })
        }
    }
}

/// Convert ay model to TY State
#[cfg(feature = "ay")]
pub(super) fn model_to_state(
    model: &tla_ay::Model,
    var_infos: &HashMap<String, VarInfo>,
    string_reverse_map: &HashMap<i64, String>,
) -> AYEnumResult<State> {
    use crate::value::FuncValue;
    use num_bigint::BigInt;

    let mut state_pairs: Vec<(Arc<str>, Value)> = Vec::new();

    for (name, info) in var_infos {
        let value = match &info.sort {
            VarSort::Bool | VarSort::Int | VarSort::String { .. } => {
                model_get_scalar(model, name, &info.sort, string_reverse_map)?
            }
            VarSort::Function { domain_keys, range } => {
                // Build function value from per-element variables
                let mut entries: Vec<(Value, Value)> = Vec::new();
                for key in domain_keys {
                    let var_name = format!("{}__{}", name, key);
                    let elem_value = model_get_scalar(model, &var_name, range, string_reverse_map)?;
                    // Convert key to Value for the domain
                    let key_value = if let Ok(n) = key.parse::<i64>() {
                        // SOUNDNESS (#633/#634): normalize to SmallInt fast path so
                        // function-domain keys match the canonical Value encoding used
                        // elsewhere (see model_get_scalar above).
                        Value::big_int(BigInt::from(n))
                    } else if key == "true" {
                        Value::Bool(true)
                    } else if key == "false" {
                        Value::Bool(false)
                    } else {
                        // Part of #3287: eager token assignment for ay model strings.
                        Value::String(intern_string(key.as_str()))
                    };
                    entries.push((key_value, elem_value));
                }
                // Sort entries by key for FuncValue constructor
                entries.sort_by(|(k1, _), (k2, _)| k1.cmp(k2));
                Value::Func(FuncValue::from_sorted_entries(entries).into())
            }
            VarSort::Tuple { element_sorts } => {
                // Build tuple value from per-element variables (1-indexed)
                let mut elements: Vec<Value> = Vec::new();
                for (i, elem_sort) in element_sorts.iter().enumerate() {
                    let var_name = format!("{}__{}", name, i + 1);
                    let elem_value =
                        model_get_scalar(model, &var_name, elem_sort, string_reverse_map)?;
                    elements.push(elem_value);
                }
                Value::Tuple(elements.into())
            }
            // Heterogeneous vars should never reach model_to_state - early error return
            VarSort::Heterogeneous { reason } => {
                return Err(AYEnumError::UnsupportedVarType {
                    var: name.clone(),
                    reason: format!("heterogeneous type in model: {}", reason),
                });
            }
        };
        state_pairs.push((info.name.clone(), value));
    }

    Ok(State::from_pairs(state_pairs))
}

/// Return the translated scalar variable names that define model blocking.
///
/// Compound TLA values are flattened by the translator into scalar AY variables
/// such as `f__key` and `tuple__1`. The blocking clause itself is asserted by
/// `tla-ay`, keeping solver-owned term construction out of the checker.
#[cfg(feature = "ay")]
pub(super) fn blocking_var_names(
    var_infos: &HashMap<String, VarInfo>,
) -> AYEnumResult<Vec<String>> {
    let mut names = Vec::new();
    for (name, info) in var_infos {
        collect_blocking_var_names(&mut names, name, &info.sort);
    }

    names.sort();
    names.dedup();

    if names.is_empty() {
        return Err(AYEnumError::TranslationFailed(
            "cannot build model blocking clause: no scalar ay variables available".to_string(),
        ));
    }

    Ok(names)
}

#[cfg(feature = "ay")]
fn collect_blocking_var_names(names: &mut Vec<String>, name: &str, sort: &VarSort) {
    match sort {
        VarSort::Bool | VarSort::Int | VarSort::String { .. } => {
            names.push(name.to_string());
        }
        VarSort::Function { domain_keys, range } => match range.as_ref() {
            VarSort::Bool | VarSort::Int | VarSort::String { .. } => {
                names.extend(domain_keys.iter().map(|key| format!("{name}__{key}")));
            }
            VarSort::Function { .. } | VarSort::Tuple { .. } | VarSort::Heterogeneous { .. } => {}
        },
        VarSort::Tuple { element_sorts } => {
            for (idx, elem_sort) in element_sorts.iter().enumerate() {
                match elem_sort {
                    VarSort::Bool | VarSort::Int | VarSort::String { .. } => {
                        names.push(format!("{}__{}", name, idx + 1));
                    }
                    VarSort::Function { .. }
                    | VarSort::Tuple { .. }
                    | VarSort::Heterogeneous { .. } => {}
                }
            }
        }
        VarSort::Heterogeneous { .. } => {}
    }
}
