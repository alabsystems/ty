// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Compound value serialization + scratch-buffer state for the JIT/AOT ABI.
//!
//! This module is the canonical source for:
//!
//! - `serialize_value` / `deserialize_value` — Convert between `tla_value::Value`
//!   and the flat `[i64]` representation used by JIT-compiled code.
//! - `infer_layout` / `infer_var_layout` — Build [`CompoundLayout`] /
//!   [`VarLayout`] descriptors from runtime values.
//! - `COMPOUND_SCRATCH_BASE`, `clear_compound_scratch`, `read_compound_scratch`,
//!   `compound_scratch_guard`, `with_compound_scratch`, `with_compound_scratch_mut`
//!   — Thread-local scratch buffer shared between trust_cg-emitted code and the
//!   interpreter fallback path.
//!
//! These functions live here so callers in `tla-check` and `tla-trust_cg` can
//! import directly from `tla_jit_abi`.
//!
//! Part of #4267 Wave 11d (epic #4251 Stage 2d).

use crate::layout::{
    CompoundLayout, SetBitmaskElement, VarLayout, TAG_BOOL, TAG_FUNC, TAG_INT, TAG_RECORD, TAG_SEQ,
    TAG_SET, TAG_STRING, TAG_TUPLE,
};
use crate::JitRuntimeError;
use tla_core::NameId;
use tla_value::value::{FuncValue, IntIntervalFunc, RecordValue, SeqValue, SortedSet, Value};
use tla_value::Rp;

// ============================================================================
// Serialization: Value -> flat i64 buffer
// ============================================================================

/// Serialize a `Value` into a flat i64 buffer, appending to `buf`.
///
/// Returns the number of i64 slots written.
///
/// The serialization format is self-describing: each value starts with a
/// type tag word, followed by type-specific payload. This allows
/// deserialization without external layout metadata (though layout
/// descriptors can validate the structure).
pub fn serialize_value(value: &Value, buf: &mut Vec<i64>) -> Result<usize, JitRuntimeError> {
    let start = buf.len();
    serialize_value_inner(value, buf)?;
    Ok(buf.len() - start)
}

/// Internal recursive serialization.
fn serialize_value_inner(value: &Value, buf: &mut Vec<i64>) -> Result<(), JitRuntimeError> {
    match value {
        Value::Bool(b) => {
            buf.push(TAG_BOOL);
            buf.push(i64::from(*b));
        }
        Value::SmallInt(n) => {
            buf.push(TAG_INT);
            buf.push(*n);
        }
        Value::Int(n) => {
            use num_traits::ToPrimitive;
            let val = n.to_i64().ok_or_else(|| {
                JitRuntimeError::CompileError(format!(
                    "BigInt value {n} does not fit in i64 for JIT serialization"
                ))
            })?;
            buf.push(TAG_INT);
            buf.push(val);
        }
        Value::String(s) => {
            let name_id = tla_core::intern_name(s);
            buf.push(TAG_STRING);
            buf.push(name_id.0 as i64);
        }
        // ModelValue is serialized identically to String: intern the name
        // and store as TAG_STRING + NameId. Both are represented as interned
        // NameId values at runtime in the JIT. Part of #3958.
        Value::ModelValue(s) => {
            let name_id = tla_core::intern_name(s);
            buf.push(TAG_STRING);
            buf.push(name_id.0 as i64);
        }
        Value::Record(rec) => {
            buf.push(TAG_RECORD);
            buf.push(rec.len() as i64);
            // iter() yields (NameId, &Value) pairs in RecordValue's canonical
            // field order (field-name string)
            for (name_id, field_val) in rec.iter() {
                buf.push(name_id.0 as i64);
                serialize_value_inner(field_val, buf)?;
            }
        }
        Value::Seq(seq) => {
            buf.push(TAG_SEQ);
            buf.push(seq.len() as i64);
            for elem in seq.iter() {
                serialize_value_inner(elem, buf)?;
            }
        }
        Value::Set(sorted_set) => {
            buf.push(TAG_SET);
            buf.push(sorted_set.len() as i64);
            for elem in sorted_set.iter() {
                serialize_value_inner(elem, buf)?;
            }
        }
        Value::Func(func) => {
            serialize_func_value(func, buf)?;
        }
        Value::IntFunc(func) => {
            serialize_int_func_value(func, buf)?;
        }
        Value::Tuple(elems) => {
            buf.push(TAG_TUPLE);
            buf.push(elems.len() as i64);
            for elem in elems.iter() {
                serialize_value_inner(elem, buf)?;
            }
        }
        _ => {
            return Err(JitRuntimeError::UnsupportedExpr(format!(
                "cannot serialize value type for JIT: {value:?}"
            )));
        }
    }
    Ok(())
}

/// Serialize a FuncValue using its overlay-aware iterator.
fn serialize_func_value(func: &FuncValue, buf: &mut Vec<i64>) -> Result<(), JitRuntimeError> {
    buf.push(TAG_FUNC);
    buf.push(func.domain_len() as i64);
    // iter() yields (key, value) pairs in domain-sorted order, overlay-aware
    for (key, val) in func.iter() {
        serialize_value_inner(key, buf)?;
        serialize_value_inner(val, buf)?;
    }
    Ok(())
}

/// Serialize an IntIntervalFunc as TAG_FUNC with integer keys [min..max].
///
/// The wire format is identical to FuncValue so deserialization always
/// produces a generic `Value::Func`. This is fine because the JIT only
/// needs the flattened i64 representation — the optimized IntFunc is an
/// interpreter-side memory optimization, not a semantic distinction.
fn serialize_int_func_value(
    func: &IntIntervalFunc,
    buf: &mut Vec<i64>,
) -> Result<(), JitRuntimeError> {
    buf.push(TAG_FUNC);
    buf.push(func.len() as i64);
    for i in 0..func.len() {
        let key = func.min() + i as i64;
        buf.push(TAG_INT);
        buf.push(key);
        serialize_value_inner(&func.values()[i], buf)?;
    }
    Ok(())
}

// ============================================================================
// Deserialization: flat i64 buffer -> Value
// ============================================================================

/// Deserialize a `Value` from a flat i64 buffer starting at `pos`.
///
/// Returns the deserialized value and the number of i64 slots consumed.
pub fn deserialize_value(buf: &[i64], pos: usize) -> Result<(Value, usize), JitRuntimeError> {
    if pos >= buf.len() {
        return Err(JitRuntimeError::CompileError(
            "buffer underflow during JIT deserialization".to_string(),
        ));
    }

    let tag = buf[pos];
    match tag {
        TAG_BOOL => {
            if pos + 1 >= buf.len() {
                return Err(JitRuntimeError::CompileError(
                    "buffer underflow reading bool value".to_string(),
                ));
            }
            Ok((Value::Bool(buf[pos + 1] != 0), 2))
        }
        TAG_INT => {
            if pos + 1 >= buf.len() {
                return Err(JitRuntimeError::CompileError(
                    "buffer underflow reading int value".to_string(),
                ));
            }
            Ok((Value::SmallInt(buf[pos + 1]), 2))
        }
        TAG_STRING => {
            if pos + 1 >= buf.len() {
                return Err(JitRuntimeError::CompileError(
                    "buffer underflow reading string value".to_string(),
                ));
            }
            let name_id = NameId(buf[pos + 1] as u32);
            let s = tla_core::resolve_name_id(name_id);
            Ok((Value::String(s.into()), 2))
        }
        TAG_RECORD => deserialize_record(buf, pos),
        TAG_SEQ => deserialize_seq(buf, pos),
        TAG_SET => deserialize_set(buf, pos),
        TAG_FUNC => deserialize_func(buf, pos),
        TAG_TUPLE => deserialize_tuple(buf, pos),
        _ => Err(JitRuntimeError::CompileError(format!(
            "unknown type tag {tag} at offset {pos} during JIT deserialization"
        ))),
    }
}

/// Deserialize a record from the flat buffer.
fn deserialize_record(buf: &[i64], pos: usize) -> Result<(Value, usize), JitRuntimeError> {
    if pos + 1 >= buf.len() {
        return Err(JitRuntimeError::CompileError(
            "buffer underflow reading record header".to_string(),
        ));
    }
    let field_count = buf[pos + 1] as usize;
    let mut offset = pos + 2;
    let mut entries = Vec::with_capacity(field_count);

    for _ in 0..field_count {
        if offset >= buf.len() {
            return Err(JitRuntimeError::CompileError(
                "buffer underflow reading record field name".to_string(),
            ));
        }
        let name_id = NameId(buf[offset] as u32);
        offset += 1;
        let (val, consumed) = deserialize_value(buf, offset)?;
        offset += consumed;
        entries.push((name_id, val));
    }

    // RecordValue expects entries in canonical field order (field-name string)
    // — they already are, because serialize preserves the RecordValue's order.
    Ok((
        Value::Record(RecordValue::from_sorted_entries(entries)),
        offset - pos,
    ))
}

/// Deserialize a sequence from the flat buffer.
fn deserialize_seq(buf: &[i64], pos: usize) -> Result<(Value, usize), JitRuntimeError> {
    if pos + 1 >= buf.len() {
        return Err(JitRuntimeError::CompileError(
            "buffer underflow reading seq header".to_string(),
        ));
    }
    let elem_count = buf[pos + 1] as usize;
    let mut offset = pos + 2;
    let mut elements = Vec::with_capacity(elem_count);

    for _ in 0..elem_count {
        let (val, consumed) = deserialize_value(buf, offset)?;
        offset += consumed;
        elements.push(val);
    }

    Ok((
        Value::Seq(Rp::new(SeqValue::from_vec(elements))),
        offset - pos,
    ))
}

/// Deserialize a set from the flat buffer.
fn deserialize_set(buf: &[i64], pos: usize) -> Result<(Value, usize), JitRuntimeError> {
    if pos + 1 >= buf.len() {
        return Err(JitRuntimeError::CompileError(
            "buffer underflow reading set header".to_string(),
        ));
    }
    let elem_count = buf[pos + 1] as usize;
    let mut offset = pos + 2;
    let mut elements = Vec::with_capacity(elem_count);

    for _ in 0..elem_count {
        let (val, consumed) = deserialize_value(buf, offset)?;
        offset += consumed;
        elements.push(val);
    }

    // Elements are already sorted because serialize iterates in canonical order.
    Ok((
        Value::Set(Rp::new(SortedSet::from_sorted_vec(elements))),
        offset - pos,
    ))
}

/// Deserialize a function from the flat buffer.
fn deserialize_func(buf: &[i64], pos: usize) -> Result<(Value, usize), JitRuntimeError> {
    if pos + 1 >= buf.len() {
        return Err(JitRuntimeError::CompileError(
            "buffer underflow reading func header".to_string(),
        ));
    }
    let pair_count = buf[pos + 1] as usize;
    let mut offset = pos + 2;
    let mut entries = Vec::with_capacity(pair_count);

    for _ in 0..pair_count {
        let (key, key_consumed) = deserialize_value(buf, offset)?;
        offset += key_consumed;
        let (val, val_consumed) = deserialize_value(buf, offset)?;
        offset += val_consumed;
        entries.push((key, val));
    }

    Ok((
        Value::Func(Rp::new(FuncValue::from_sorted_entries(entries))),
        offset - pos,
    ))
}

/// Deserialize a tuple from the flat buffer.
fn deserialize_tuple(buf: &[i64], pos: usize) -> Result<(Value, usize), JitRuntimeError> {
    if pos + 1 >= buf.len() {
        return Err(JitRuntimeError::CompileError(
            "buffer underflow reading tuple header".to_string(),
        ));
    }
    let elem_count = buf[pos + 1] as usize;
    let mut offset = pos + 2;
    let mut elements = Vec::with_capacity(elem_count);

    for _ in 0..elem_count {
        let (val, consumed) = deserialize_value(buf, offset)?;
        offset += consumed;
        elements.push(val);
    }

    Ok((Value::Tuple(elements.into()), offset - pos))
}

// ============================================================================
// Layout inference
// ============================================================================

/// Infer a `CompoundLayout` from a runtime `Value`.
///
/// This is useful when the type of a state variable is not statically known
/// and must be determined from the initial state.
pub fn infer_layout(value: &Value) -> CompoundLayout {
    match value {
        Value::Bool(_) => CompoundLayout::Bool,
        Value::SmallInt(_) | Value::Int(_) => CompoundLayout::Int,
        Value::String(_) | Value::ModelValue(_) => CompoundLayout::String,
        Value::Record(rec) => {
            let fields = rec
                .iter()
                .map(|(nid, val)| (nid, infer_layout(val)))
                .collect();
            CompoundLayout::Record { fields }
        }
        Value::Seq(seq) => {
            let element_layout = seq
                .get(0)
                .map(infer_layout)
                .unwrap_or(CompoundLayout::Dynamic);
            CompoundLayout::Sequence {
                element_layout: Box::new(element_layout),
                element_count: Some(seq.len()),
                // Inferred from a single concrete value: an observed length, not
                // a proven upper bound. Fail-closed for capacity-driven domain
                // enumeration.
                capacity_proven: false,
            }
        }
        Value::Set(sorted_set) => {
            let element_layout = sorted_set
                .iter()
                .next()
                .map(infer_layout)
                .unwrap_or(CompoundLayout::Dynamic);
            CompoundLayout::Set {
                element_layout: Box::new(element_layout),
                element_count: Some(sorted_set.len()),
            }
        }
        Value::Func(func) => {
            let key_layout = func
                .domain_iter()
                .next()
                .map(infer_layout)
                .unwrap_or(CompoundLayout::Dynamic);
            let value_layout = if func.domain_is_empty() {
                CompoundLayout::Dynamic
            } else {
                infer_layout(func.get_value_at(0))
            };

            // Detect contiguous integer domain for direct-index optimization.
            // Part of #3985: Phase 2 compound layout wiring.
            let domain_lo = if matches!(key_layout, CompoundLayout::Int) && !func.domain_is_empty()
            {
                let mut min_key = i64::MAX;
                let mut max_key = i64::MIN;
                let mut all_int = true;
                for key in func.domain_iter() {
                    match key {
                        Value::SmallInt(n) => {
                            min_key = min_key.min(*n);
                            max_key = max_key.max(*n);
                        }
                        _ => {
                            all_int = false;
                            break;
                        }
                    }
                }
                if all_int {
                    let expected_len = (max_key - min_key + 1) as usize;
                    if expected_len == func.domain_len() {
                        Some(min_key)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let key_layout = if domain_lo.is_none() {
                explicit_scalar_domain_key_layout(&key_layout, func.domain_iter())
                    .unwrap_or(key_layout)
            } else {
                key_layout
            };

            CompoundLayout::Function {
                key_layout: Box::new(key_layout),
                value_layout: Box::new(value_layout),
                pair_count: Some(func.domain_len()),
                domain_lo,
            }
        }
        Value::IntFunc(func) => {
            let value_layout = func
                .values()
                .first()
                .map(infer_layout)
                .unwrap_or(CompoundLayout::Dynamic);
            CompoundLayout::Function {
                key_layout: Box::new(CompoundLayout::Int),
                value_layout: Box::new(value_layout),
                pair_count: Some(func.len()),
                domain_lo: Some(IntIntervalFunc::min(func)),
            }
        }
        Value::Tuple(elems) => {
            let element_layouts = elems.iter().map(infer_layout).collect();
            CompoundLayout::Tuple { element_layouts }
        }
        _ => CompoundLayout::Dynamic,
    }
}

fn explicit_scalar_domain_key_layout<'a, I>(
    key_layout: &CompoundLayout,
    keys: I,
) -> Option<CompoundLayout>
where
    I: IntoIterator<Item = &'a Value>,
{
    let keys: Option<Vec<SetBitmaskElement>> = keys
        .into_iter()
        .map(|key| scalar_domain_element_for_layout(key_layout, key))
        .collect();
    let keys = keys?;
    Some(CompoundLayout::ExplicitScalarDomain {
        key_layout: Box::new(key_layout.clone()),
        keys,
    })
}

fn scalar_domain_element_for_layout(
    key_layout: &CompoundLayout,
    key: &Value,
) -> Option<SetBitmaskElement> {
    match (key_layout, key) {
        (CompoundLayout::Int, Value::SmallInt(n)) => Some(SetBitmaskElement::Int(*n)),
        (CompoundLayout::Int, Value::Int(n)) => {
            use num_traits::ToPrimitive;
            n.to_i64().map(SetBitmaskElement::Int)
        }
        (CompoundLayout::Bool, Value::Bool(b)) => Some(SetBitmaskElement::Bool(*b)),
        (CompoundLayout::String, Value::String(name)) => Some(SetBitmaskElement::String(
            tla_core::intern_name(name.as_ref()),
        )),
        (CompoundLayout::String, Value::ModelValue(name)) => Some(SetBitmaskElement::ModelValue(
            tla_core::intern_name(name.as_ref()),
        )),
        _ => None,
    }
}

// ============================================================================
// VarLayout inference from Value
// ============================================================================

/// Infer a `VarLayout` from a runtime `Value`.
///
/// Returns `ScalarInt`/`ScalarBool` for scalar values, or `Compound(..)` for
/// compound values.
pub fn infer_var_layout(value: &Value) -> VarLayout {
    match value {
        Value::SmallInt(_) | Value::Int(_) => VarLayout::ScalarInt,
        Value::Bool(_) => VarLayout::ScalarBool,
        _ => VarLayout::Compound(infer_layout(value)),
    }
}

// ============================================================================
// Compound scratch buffer (thread-local, shared between JIT + interpreter)
// ============================================================================

/// Sentinel base offset for compound scratch buffer references.
///
/// A JIT-constructed compound value writes itself into the thread-local
/// scratch buffer and returns `COMPOUND_SCRATCH_BASE + start_pos`, allowing
/// the interpreter fallback to detect "compound was constructed here" and
/// deserialize via [`read_compound_scratch`].
pub const COMPOUND_SCRATCH_BASE: i64 = 0x7FFF_0000_0000_0000_u64 as i64;

thread_local! {
    /// Thread-local scratch buffer for compiled compound values.
    ///
    /// Shared between trust_cg-emitted native code and the interpreter fallback
    /// (in `tla-check::check::model_checker::invariants::eval`). Exposed via
    /// [`with_compound_scratch`] / [`with_compound_scratch_mut`].
    static COMPOUND_SCRATCH: std::cell::RefCell<Vec<i64>> =
        std::cell::RefCell::new(Vec::with_capacity(64));
}

/// Clear the compound scratch buffer before each action evaluation.
pub fn clear_compound_scratch() {
    COMPOUND_SCRATCH.with(|buf| buf.borrow_mut().clear());
}

/// RAII guard that clears the compound scratch buffer on drop.
pub struct CompoundScratchGuard;

impl Drop for CompoundScratchGuard {
    fn drop(&mut self) {
        COMPOUND_SCRATCH.with(|buf| buf.borrow_mut().clear());
    }
}

/// Acquire a guard that will clear the compound scratch buffer when dropped.
#[must_use]
pub fn compound_scratch_guard() -> CompoundScratchGuard {
    clear_compound_scratch();
    CompoundScratchGuard
}

/// Read from the compound scratch buffer.
pub fn read_compound_scratch() -> Vec<i64> {
    COMPOUND_SCRATCH.with(|buf| buf.borrow().clone())
}

/// Access the compound scratch buffer for read-only operations.
///
/// Used by JIT runtime helpers that need to inspect the buffer without
/// allocating a copy.
pub fn with_compound_scratch<F, R>(f: F) -> R
where
    F: FnOnce(&Vec<i64>) -> R,
{
    COMPOUND_SCRATCH.with(|buf| f(&buf.borrow()))
}

/// Access the compound scratch buffer for mutation.
///
/// Used by JIT runtime helpers (`jit_record_new_scalar`, `jit_seq_tail`, etc.)
/// that construct compound values and append them to the shared scratch buffer.
pub fn with_compound_scratch_mut<F, R>(f: F) -> R
where
    F: FnOnce(&mut Vec<i64>) -> R,
{
    COMPOUND_SCRATCH.with(|buf| f(&mut buf.borrow_mut()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // ========================================================================
    // Scalar round-trips
    // ========================================================================

    #[test]
    fn test_roundtrip_bool_true() {
        let val = Value::Bool(true);
        let mut buf = Vec::new();
        let written = serialize_value(&val, &mut buf).expect("serialize bool");
        assert_eq!(written, 2);
        assert_eq!(buf, vec![TAG_BOOL, 1]);

        let (deserialized, consumed) = deserialize_value(&buf, 0).expect("deserialize bool");
        assert_eq!(consumed, 2);
        assert_eq!(deserialized, Value::Bool(true));
    }

    #[test]
    fn test_roundtrip_int_scalar() {
        let val = Value::SmallInt(42);
        let mut buf = Vec::new();
        let written = serialize_value(&val, &mut buf).expect("serialize int");
        assert_eq!(written, 2);
        assert_eq!(buf, vec![TAG_INT, 42]);

        let (deserialized, consumed) = deserialize_value(&buf, 0).expect("deserialize int");
        assert_eq!(consumed, 2);
        assert_eq!(deserialized, Value::SmallInt(42));
    }

    #[test]
    fn test_roundtrip_string_scalar() {
        let val = Value::String(Rp::from("hello"));
        let mut buf = Vec::new();
        serialize_value(&val, &mut buf).expect("serialize string");
        let (deserialized, _) = deserialize_value(&buf, 0).expect("deserialize string");
        match &deserialized {
            Value::String(s) => assert_eq!(&**s, "hello"),
            other => panic!("expected String, got {other:?}"),
        }
    }

    #[test]
    fn test_roundtrip_model_value_serializes_as_string() {
        // ModelValue is serialized identically to String (TAG_STRING + NameId),
        // so a round-trip intentionally collapses to a String value. Pinning
        // this documented lossy behaviour (Part of #3958).
        let val = Value::ModelValue(Rp::from("proc1"));
        let mut buf = Vec::new();
        let written = serialize_value(&val, &mut buf).expect("serialize model value");
        assert_eq!(written, 2);
        assert_eq!(buf[0], TAG_STRING);

        let (deserialized, consumed) = deserialize_value(&buf, 0).expect("deserialize");
        assert_eq!(consumed, 2);
        assert_eq!(deserialized, Value::String(Rp::from("proc1")));
    }

    // ========================================================================
    // Compound round-trips — the recursive serialize/deserialize path
    // ========================================================================

    /// Assert that a value survives serialize -> deserialize unchanged and that
    /// the slot count reported by `serialize_value` equals the slot count
    /// consumed by `deserialize_value`.
    fn assert_roundtrip(val: &Value) {
        let mut buf = Vec::new();
        let written = serialize_value(val, &mut buf).expect("serialize");
        assert_eq!(written, buf.len(), "written slots must equal buffer length");
        let (deserialized, consumed) = deserialize_value(&buf, 0).expect("deserialize");
        assert_eq!(
            consumed, written,
            "consumed slots must equal written slots for {val:?}"
        );
        assert_eq!(&deserialized, val, "round-trip mismatch for {val:?}");
    }

    #[test]
    fn test_roundtrip_record_with_mixed_fields() {
        let rec = Value::Record(RecordValue::from_sorted_entries(vec![
            (tla_core::intern_name("a"), Value::SmallInt(1)),
            (tla_core::intern_name("b"), Value::Bool(true)),
        ]));
        assert_roundtrip(&rec);
    }

    #[test]
    fn test_roundtrip_sequence_of_ints() {
        let seq = Value::seq([Value::SmallInt(3), Value::SmallInt(7), Value::SmallInt(-2)]);
        assert_roundtrip(&seq);
    }

    #[test]
    fn test_roundtrip_set_of_ints() {
        let set = Value::set([Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)]);
        assert_roundtrip(&set);
    }

    #[test]
    fn test_roundtrip_tuple_mixed() {
        let tuple = Value::tuple([Value::SmallInt(9), Value::Bool(false)]);
        assert_roundtrip(&tuple);
    }

    #[test]
    fn test_roundtrip_function_serializes_as_func() {
        // Func keys/values are stored as tagged pairs; a contiguous integer
        // domain still round-trips back into a generic Value::Func.
        let func = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
            (Value::SmallInt(0), Value::Bool(false)),
            (Value::SmallInt(1), Value::Bool(true)),
        ])));
        assert_roundtrip(&func);
    }

    #[test]
    fn test_roundtrip_int_func_collapses_to_generic_func() {
        // IntFunc serializes with the identical wire format as Func, so a
        // round-trip produces a generic Value::Func with explicit int keys
        // reconstructed from the [min..max] interval.
        let int_func = Value::IntFunc(Rp::new(IntIntervalFunc::new(
            2,
            3,
            vec![Value::SmallInt(10), Value::SmallInt(20)],
        )));
        let mut buf = Vec::new();
        serialize_value(&int_func, &mut buf).expect("serialize int func");
        // First slot is TAG_FUNC, not a dedicated int-func tag.
        assert_eq!(buf[0], TAG_FUNC);
        let (deserialized, _) = deserialize_value(&buf, 0).expect("deserialize");
        let expected = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
            (Value::SmallInt(2), Value::SmallInt(10)),
            (Value::SmallInt(3), Value::SmallInt(20)),
        ])));
        assert_eq!(deserialized, expected);
    }

    #[test]
    fn test_roundtrip_nested_record_in_seq() {
        // Exercises recursion through two compound layers.
        let nested = Value::seq([
            Value::Record(RecordValue::from_sorted_entries(vec![(
                tla_core::intern_name("x"),
                Value::SmallInt(5),
            )])),
            Value::Record(RecordValue::from_sorted_entries(vec![(
                tla_core::intern_name("x"),
                Value::SmallInt(6),
            )])),
        ]);
        assert_roundtrip(&nested);
    }

    #[test]
    fn test_roundtrip_empty_collections() {
        assert_roundtrip(&Value::seq([]));
        assert_roundtrip(&Value::set([]));
        assert_roundtrip(&Value::tuple([]));
        assert_roundtrip(&Value::Record(RecordValue::from_sorted_entries(vec![])));
    }

    #[test]
    fn test_serialize_appends_without_clobbering_prefix() {
        // serialize_value appends; the returned count is the number of NEW
        // slots, and any pre-existing prefix is preserved.
        let mut buf = vec![100, 200];
        let written = serialize_value(&Value::SmallInt(7), &mut buf).expect("serialize");
        assert_eq!(written, 2);
        assert_eq!(buf, vec![100, 200, TAG_INT, 7]);
        // Deserialize from the offset where we appended.
        let (val, consumed) = deserialize_value(&buf, 2).expect("deserialize at offset");
        assert_eq!(consumed, 2);
        assert_eq!(val, Value::SmallInt(7));
    }

    // ========================================================================
    // Serialization error paths
    // ========================================================================

    #[test]
    fn test_serialize_unsupported_value_is_error() {
        // AnySet is not a finite serializable value; serialize must reject it.
        let err = serialize_value(&Value::AnySet, &mut Vec::new())
            .expect_err("non-finite value must fail to serialize");
        assert!(
            matches!(err, JitRuntimeError::UnsupportedExpr(_)),
            "expected UnsupportedExpr, got {err:?}"
        );
    }

    #[test]
    fn test_serialize_unsupported_value_inside_compound_propagates() {
        // An unsupported element nested inside a serializable container must
        // still surface the error rather than silently dropping the element.
        let seq = Value::seq([Value::SmallInt(1), Value::AnySet]);
        let err = serialize_value(&seq, &mut Vec::new())
            .expect_err("unsupported nested element must fail");
        assert!(matches!(err, JitRuntimeError::UnsupportedExpr(_)));
    }

    // ========================================================================
    // Deserialization error paths (documented buffer-underflow / bad-tag)
    // ========================================================================

    #[test]
    fn test_deserialize_empty_buffer_is_error() {
        let err = deserialize_value(&[], 0).expect_err("empty buffer must fail");
        assert!(matches!(err, JitRuntimeError::CompileError(_)));
    }

    #[test]
    fn test_deserialize_pos_past_end_is_error() {
        let buf = [TAG_INT, 5];
        let err = deserialize_value(&buf, 2).expect_err("pos at len must fail");
        assert!(matches!(err, JitRuntimeError::CompileError(_)));
    }

    #[test]
    fn test_deserialize_unknown_tag_is_error() {
        // 999 is not any TAG_* constant.
        let buf = [999i64, 0];
        let err = deserialize_value(&buf, 0).expect_err("unknown tag must fail");
        match err {
            JitRuntimeError::CompileError(msg) => {
                assert!(msg.contains("unknown type tag"), "got: {msg}");
            }
            other => panic!("expected CompileError, got {other:?}"),
        }
    }

    #[test]
    fn test_deserialize_scalar_missing_payload_is_error() {
        // A lone scalar tag with no following payload word is a truncated
        // buffer and must be rejected for every scalar tag.
        for tag in [TAG_INT, TAG_BOOL, TAG_STRING] {
            let result = deserialize_value(&[tag], 0);
            let err = result.expect_err("scalar tag with no payload should error");
            assert!(
                matches!(err, JitRuntimeError::CompileError(_)),
                "tag {tag} should yield CompileError, got {err:?}"
            );
        }
    }

    #[test]
    fn test_deserialize_record_header_truncated_is_error() {
        // TAG_RECORD with no field-count word.
        let err = deserialize_value(&[TAG_RECORD], 0).expect_err("truncated record header");
        assert!(matches!(err, JitRuntimeError::CompileError(_)));
    }

    #[test]
    fn test_deserialize_record_truncated_field_is_error() {
        // Claims one field but the buffer ends before the field name slot.
        let buf = [TAG_RECORD, 1];
        let err = deserialize_value(&buf, 0).expect_err("truncated record field");
        assert!(matches!(err, JitRuntimeError::CompileError(_)));
    }

    #[test]
    fn test_deserialize_seq_truncated_element_is_error() {
        // Claims two elements but provides only one complete element.
        let buf = [TAG_SEQ, 2, TAG_INT, 1];
        let err = deserialize_value(&buf, 0).expect_err("truncated seq element");
        assert!(matches!(err, JitRuntimeError::CompileError(_)));
    }

    #[test]
    fn test_deserialize_func_truncated_pair_is_error() {
        // Claims one (key, value) pair but only the key is present.
        let buf = [TAG_FUNC, 1, TAG_INT, 0];
        let err = deserialize_value(&buf, 0).expect_err("truncated func pair");
        assert!(matches!(err, JitRuntimeError::CompileError(_)));
    }

    // ========================================================================
    // Layout inference for compound shapes
    // ========================================================================

    #[test]
    fn test_infer_var_layout_compound_for_seq() {
        let seq = Value::seq([Value::SmallInt(1)]);
        match infer_var_layout(&seq) {
            VarLayout::Compound(CompoundLayout::Sequence {
                element_count,
                capacity_proven,
                ..
            }) => {
                assert_eq!(element_count, Some(1));
                // Inferred from a single concrete value: observed, not proven.
                assert!(!capacity_proven);
            }
            other => panic!("expected Compound(Sequence), got {other:?}"),
        }
    }

    #[test]
    fn test_infer_layout_set_records_element_layout_and_count() {
        let set = Value::set([Value::SmallInt(1), Value::SmallInt(2)]);
        match infer_layout(&set) {
            CompoundLayout::Set {
                element_layout,
                element_count,
            } => {
                assert_eq!(*element_layout, CompoundLayout::Int);
                assert_eq!(element_count, Some(2));
            }
            other => panic!("expected Set, got {other:?}"),
        }
    }

    #[test]
    fn test_infer_layout_tuple_preserves_per_element_layouts() {
        let tuple = Value::tuple([Value::SmallInt(1), Value::Bool(true)]);
        assert_eq!(
            infer_layout(&tuple),
            CompoundLayout::Tuple {
                element_layouts: vec![CompoundLayout::Int, CompoundLayout::Bool],
            }
        );
    }

    #[test]
    fn test_infer_layout_empty_seq_uses_dynamic_element_layout() {
        match infer_layout(&Value::seq([])) {
            CompoundLayout::Sequence {
                element_layout,
                element_count,
                ..
            } => {
                assert_eq!(*element_layout, CompoundLayout::Dynamic);
                assert_eq!(element_count, Some(0));
            }
            other => panic!("expected Sequence, got {other:?}"),
        }
    }

    #[test]
    fn test_infer_layout_contiguous_int_domain_func_sets_domain_lo() {
        // A function over a contiguous integer domain {2,3,4} must be detected
        // as a direct-index array with domain_lo = 2 (Part of #3985).
        let func = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
            (Value::SmallInt(2), Value::Bool(false)),
            (Value::SmallInt(3), Value::Bool(true)),
            (Value::SmallInt(4), Value::Bool(false)),
        ])));
        match infer_layout(&func) {
            CompoundLayout::Function {
                key_layout,
                value_layout,
                pair_count,
                domain_lo,
            } => {
                assert_eq!(*key_layout, CompoundLayout::Int);
                assert_eq!(*value_layout, CompoundLayout::Bool);
                assert_eq!(pair_count, Some(3));
                assert_eq!(domain_lo, Some(2));
            }
            other => panic!("expected Function, got {other:?}"),
        }
    }

    #[test]
    fn test_infer_layout_sparse_int_domain_func_has_no_domain_lo() {
        // A non-contiguous integer domain {0,2} cannot use direct indexing, so
        // domain_lo must be None and the keys must fall back to an explicit
        // scalar domain.
        let func = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
            (Value::SmallInt(0), Value::Bool(false)),
            (Value::SmallInt(2), Value::Bool(true)),
        ])));
        match infer_layout(&func) {
            CompoundLayout::Function {
                domain_lo,
                key_layout,
                ..
            } => {
                assert_eq!(domain_lo, None);
                assert!(
                    matches!(*key_layout, CompoundLayout::ExplicitScalarDomain { .. }),
                    "sparse int domain should fall back to ExplicitScalarDomain, got {key_layout:?}"
                );
            }
            other => panic!("expected Function, got {other:?}"),
        }
    }

    // ========================================================================
    // Scratch buffer read-only accessor
    // ========================================================================

    #[test]
    fn test_with_compound_scratch_read_only_sees_mutations() {
        let _guard = compound_scratch_guard();
        with_compound_scratch_mut(|buf| buf.extend_from_slice(&[7, 8, 9]));
        let observed = with_compound_scratch(|buf| buf.clone());
        assert_eq!(observed, vec![7, 8, 9]);
    }

    #[test]
    fn test_compound_scratch_base_sentinel() {
        // Sentinel must be high enough to avoid collision with legitimate
        // serialization offsets (which are small usize values).
        const { assert!(COMPOUND_SCRATCH_BASE > 0) };
        const { assert!(COMPOUND_SCRATCH_BASE > u32::MAX as i64) };
    }

    #[test]
    fn test_compound_scratch_clear() {
        with_compound_scratch_mut(|buf| buf.push(123));
        assert_eq!(read_compound_scratch(), vec![123]);
        clear_compound_scratch();
        assert_eq!(read_compound_scratch(), Vec::<i64>::new());
    }

    #[test]
    fn test_compound_scratch_guard_clears() {
        with_compound_scratch_mut(|buf| buf.push(1));
        {
            let _guard = compound_scratch_guard();
            with_compound_scratch_mut(|buf| buf.push(2));
            assert_eq!(read_compound_scratch(), vec![2]);
        }
        assert_eq!(read_compound_scratch(), Vec::<i64>::new());
    }

    #[test]
    fn test_infer_var_layout_scalar_int() {
        assert_eq!(infer_var_layout(&Value::SmallInt(0)), VarLayout::ScalarInt);
    }

    #[test]
    fn test_infer_var_layout_scalar_bool() {
        assert_eq!(infer_var_layout(&Value::Bool(true)), VarLayout::ScalarBool);
    }

    #[test]
    fn test_infer_layout_model_value_function_domain_is_explicit() {
        let value = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
            (Value::ModelValue(Rp::from("p1")), Value::Bool(false)),
            (Value::ModelValue(Rp::from("p2")), Value::Bool(true)),
        ])));

        assert_eq!(
            infer_layout(&value),
            CompoundLayout::Function {
                key_layout: Box::new(CompoundLayout::ExplicitScalarDomain {
                    key_layout: Box::new(CompoundLayout::String),
                    keys: vec![
                        SetBitmaskElement::ModelValue(tla_core::intern_name("p1")),
                        SetBitmaskElement::ModelValue(tla_core::intern_name("p2")),
                    ],
                }),
                value_layout: Box::new(CompoundLayout::Bool),
                pair_count: Some(2),
                domain_lo: None,
            }
        );
    }
}
