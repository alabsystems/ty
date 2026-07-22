// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared `Value` to JIT scalar helpers.
//!
//! These helpers convert between `tla_value::Value` scalar forms and the raw
//! i64 register representation used by compiled code. They live in
//! `tla-jit-abi` so model-checker dispatch code can keep using the stable ABI
//! crate without depending on the trust-codegen implementation crate.
//!
//! Part of #4395.

use num_traits::ToPrimitive;
use std::fmt::Write as _;
use tla_value::Value;

use crate::{specialized_key, SetBitmaskElement, SpecType};

/// Convert a scalar [`Value`] to its compiled i64 register representation.
#[must_use]
pub fn value_to_jit_i64(value: &Value) -> Option<i64> {
    match value {
        Value::SmallInt(n) => Some(*n),
        Value::Int(n) => n.to_i64(),
        Value::Bool(b) => Some(i64::from(*b)),
        Value::String(s) | Value::ModelValue(s) => {
            let name_id = tla_core::intern_name(s);
            Some(i64::from(name_id.0))
        }
        _ => None,
    }
}

/// Convert a slice of `(name, Value)` bindings to compiled i64 values.
///
/// Returns `Some(Vec<i64>)` only when every binding value is scalar enough for
/// specialization.
#[must_use]
pub fn bindings_to_jit_i64(bindings: &[(std::sync::Arc<str>, Value)]) -> Option<Vec<i64>> {
    bindings
        .iter()
        .map(|(_, val)| value_to_jit_i64(val))
        .collect()
}

/// Return whether a value can be embedded as a typed binding literal.
///
/// Scalars must fit the existing i64 ABI. Finite compound values are accepted
/// when every nested value can also be materialized through `LoadConst`.
#[must_use]
pub fn value_is_finite_binding_literal(value: &Value) -> bool {
    typed_binding_value_key_fragment(value).is_some()
}

/// Return whether every value can be embedded as a typed binding literal.
#[must_use]
pub fn values_are_finite_binding_literals(values: &[Value]) -> bool {
    values.iter().all(value_is_finite_binding_literal)
}

/// Convert one exact compact-set universe element back to a typed TLA+ value.
#[must_use]
pub fn set_bitmask_element_to_value(element: SetBitmaskElement) -> Value {
    match element {
        SetBitmaskElement::Int(value) => Value::SmallInt(value),
        SetBitmaskElement::Bool(value) => Value::Bool(value),
        SetBitmaskElement::String(name) => Value::String(tla_core::resolve_name_id(name).into()),
        SetBitmaskElement::ModelValue(name) => Value::ModelValue(tla_core::resolve_name_id(name).into()),
    }
}

/// Build a finite set literal from exact compact-set universe elements.
#[must_use]
pub fn set_bitmask_elements_to_value(elements: &[SetBitmaskElement]) -> Value {
    Value::set(elements.iter().copied().map(set_bitmask_element_to_value))
}

/// Construct the executable binding key for typed literal values.
///
/// Scalar-only values intentionally keep the historical raw-i64 key format:
/// `Action__1_2`. The typed structural form is used only when at least one
/// finite compound literal is present.
#[must_use]
pub fn binding_key_for_values(action_name: &str, values: &[Value]) -> Option<String> {
    if let Some(raw_values) = values
        .iter()
        .map(value_to_jit_i64)
        .collect::<Option<Vec<_>>>()
    {
        return Some(specialized_key(action_name, &raw_values));
    }

    let mut key = String::with_capacity(action_name.len() + 16 + values.len() * 8);
    key.push_str(action_name);
    key.push_str("__typed");
    for value in values {
        let fragment = typed_binding_value_key_fragment(value)?;
        write!(&mut key, "_{}:", fragment.len()).ok()?;
        key.push_str(&fragment);
    }
    Some(key)
}

/// Construct the executable binding key for split-action binding metadata.
#[must_use]
pub fn binding_key_for_bindings(
    action_name: &str,
    bindings: &[(std::sync::Arc<str>, Value)],
) -> Option<String> {
    let values = bindings
        .iter()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    binding_key_for_values(action_name, &values)
}

fn typed_binding_value_key_fragment(value: &Value) -> Option<String> {
    let mut out = String::new();
    push_typed_binding_value_key(&mut out, value)?;
    Some(out)
}

fn push_typed_part(out: &mut String, value: &Value) -> Option<()> {
    let mut part = String::new();
    push_typed_binding_value_key(&mut part, value)?;
    write!(out, "{}:", part.len()).ok()?;
    out.push_str(&part);
    out.push(';');
    Some(())
}

fn push_typed_binding_value_key(out: &mut String, value: &Value) -> Option<()> {
    match value {
        Value::SmallInt(value) => {
            write!(out, "i{value}").ok()?;
        }
        Value::Int(value) => {
            let value = value.to_i64()?;
            write!(out, "i{value}").ok()?;
        }
        Value::Bool(value) => {
            out.push_str(if *value { "b1" } else { "b0" });
        }
        Value::String(value) => {
            let name_id = tla_core::intern_name(value.as_ref());
            write!(out, "s{}", name_id.0).ok()?;
        }
        Value::ModelValue(value) => {
            let name_id = tla_core::intern_name(value.as_ref());
            write!(out, "m{}", name_id.0).ok()?;
        }
        Value::Set(set) => {
            write!(out, "set{}[", set.len()).ok()?;
            for elem in set.iter() {
                push_typed_part(out, elem)?;
            }
            out.push(']');
        }
        Value::Interval(interval) => {
            write!(out, "interval{}..{}", interval.low(), interval.high()).ok()?;
        }
        Value::Seq(seq) => {
            write!(out, "seq{}[", seq.len()).ok()?;
            for elem in seq.iter() {
                push_typed_part(out, elem)?;
            }
            out.push(']');
        }
        Value::Tuple(tuple) => {
            write!(out, "tuple{}[", tuple.len()).ok()?;
            for elem in tuple.iter() {
                push_typed_part(out, elem)?;
            }
            out.push(']');
        }
        Value::Record(record) => {
            write!(out, "record{}[", record.len()).ok()?;
            for (field, field_value) in record.iter() {
                write!(out, "{}=", field.0).ok()?;
                push_typed_part(out, field_value)?;
            }
            out.push(']');
        }
        Value::Func(func) => {
            write!(out, "func{}[", func.domain_len()).ok()?;
            for (key, mapped_value) in func.iter() {
                push_typed_part(out, key)?;
                out.push('=');
                push_typed_part(out, mapped_value)?;
            }
            out.push(']');
        }
        Value::IntFunc(func) => {
            write!(
                out,
                "ifunc{}..{}[",
                func.as_ref().min(),
                func.as_ref().max()
            )
            .ok()?;
            for mapped_value in func.values() {
                push_typed_part(out, mapped_value)?;
            }
            out.push(']');
        }
        _ => return None,
    }
    Some(())
}

/// Classify a concrete runtime value into a specialization-friendly type.
#[must_use]
pub fn classify_value(value: &Value) -> SpecType {
    match value {
        Value::SmallInt(_) => SpecType::Int,
        Value::Bool(_) => SpecType::Bool,
        Value::String(_) => SpecType::String,
        Value::Set(_) => SpecType::FiniteSet,
        Value::Record(_) => SpecType::Record,
        Value::Seq(_) => SpecType::Seq,
        Value::Func(_) | Value::IntFunc(_) => SpecType::Func,
        Value::Tuple(_) => SpecType::Tuple,
        _ => SpecType::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_to_jit_i64_scalar_values() {
        assert_eq!(value_to_jit_i64(&Value::SmallInt(42)), Some(42));
        assert_eq!(value_to_jit_i64(&Value::SmallInt(-1)), Some(-1));
        assert_eq!(value_to_jit_i64(&Value::Bool(true)), Some(1));
        assert_eq!(value_to_jit_i64(&Value::Bool(false)), Some(0));
    }

    #[test]
    fn value_to_jit_i64_compound_returns_none() {
        let seq = Value::seq([Value::SmallInt(1), Value::SmallInt(2)]);
        assert_eq!(value_to_jit_i64(&seq), None);
    }

    #[test]
    fn bindings_to_jit_i64_rejects_any_compound_binding() {
        let bindings: Vec<(std::sync::Arc<str>, Value)> = vec![
            (std::sync::Arc::from("i"), Value::SmallInt(3)),
            (std::sync::Arc::from("s"), Value::seq([Value::SmallInt(1)])),
        ];
        assert_eq!(bindings_to_jit_i64(&bindings), None);
    }

    #[test]
    fn bindings_to_jit_i64_preserves_scalar_order() {
        let bindings: Vec<(std::sync::Arc<str>, Value)> = vec![
            (std::sync::Arc::from("i"), Value::SmallInt(3)),
            (std::sync::Arc::from("b"), Value::Bool(true)),
        ];
        assert_eq!(bindings_to_jit_i64(&bindings), Some(vec![3, 1]));
    }

    #[test]
    fn binding_key_for_values_preserves_scalar_key_format() {
        assert_eq!(
            binding_key_for_values("SendMsg", &[Value::SmallInt(3), Value::Bool(true)]),
            Some("SendMsg__3_1".to_string())
        );
    }

    #[test]
    fn binding_key_for_values_accepts_typed_finite_compound_literals() {
        let key = binding_key_for_values(
            "UseSet",
            &[Value::set([Value::SmallInt(2), Value::SmallInt(1)])],
        )
        .expect("finite set literal should produce a typed key");

        assert!(key.starts_with("UseSet__typed"));
        assert!(values_are_finite_binding_literals(&[Value::set([
            Value::SmallInt(1),
            Value::SmallInt(2),
        ])]));
    }

    #[test]
    fn binding_key_for_values_preserves_model_value_set_elements() {
        let key = binding_key_for_values(
            "UseResources",
            &[Value::set([Value::ModelValue("r1".into())])],
        )
        .expect("finite model-value set literal should produce a typed key");
        let string_key =
            binding_key_for_values("UseResources", &[Value::set([Value::String("r1".into())])])
                .expect("finite string set literal should produce a typed key");

        assert!(
            key.contains('m'),
            "model-value fragment should be typed: {key}"
        );
        assert_ne!(
            key, string_key,
            "typed finite-set keys must distinguish strings from model values"
        );
    }

    #[test]
    fn set_bitmask_elements_to_value_preserves_model_value_kind() {
        let r1 = tla_core::intern_name("r1");
        let value = set_bitmask_elements_to_value(&[SetBitmaskElement::ModelValue(r1)]);

        let Value::Set(ref set) = value else {
            panic!("exact element list should convert to a finite set");
        };
        assert!(set
            .iter()
            .any(|value| { matches!(value, Value::ModelValue(name) if name.as_ref() == "r1") }));
    }

    #[test]
    fn binding_key_for_values_rejects_non_finite_literals() {
        assert_eq!(binding_key_for_values("Any", &[Value::AnySet]), None);
    }

    #[test]
    fn classify_value_maps_major_variants() {
        assert_eq!(classify_value(&Value::SmallInt(0)), SpecType::Int);
        assert_eq!(classify_value(&Value::Bool(true)), SpecType::Bool);
        assert_eq!(classify_value(&Value::string("s")), SpecType::String);
        assert_eq!(
            classify_value(&Value::set([Value::SmallInt(1)])),
            SpecType::FiniteSet
        );
        assert_eq!(
            classify_value(&Value::seq([Value::SmallInt(1)])),
            SpecType::Seq
        );
        assert_eq!(
            classify_value(&Value::tuple([Value::SmallInt(1)])),
            SpecType::Tuple
        );
    }
}
