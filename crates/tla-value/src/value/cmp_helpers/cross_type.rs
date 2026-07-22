// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::super::{BagValue, FuncValue, IntIntervalFunc, RecordValue, Value};
use super::primitives::{cmp_i64_with_value, cmp_str_with_value, eq_i64_with_value, type_order};
use std::cmp::Ordering;

// Helper to compare a compact Bag with a FuncValue without materializing.
//
// SOUNDNESS: MUST produce the same ordering as FuncValue::cmp on the bag's
// materialized general form. Bag entries are (elem, SmallInt(count)) in the
// same Value::cmp-sorted order as the equivalent Func's entries, so the
// entry-wise lexicographic zip below is exactly FuncValue::cmp.
pub(super) fn cmp_bag_with_func(bag: &BagValue, func: &FuncValue) -> Ordering {
    for ((bag_key, bag_count), (func_key, func_val)) in bag.entries().zip(func.mapping_iter()) {
        match bag_key.cmp(func_key) {
            Ordering::Equal => {}
            ord => return ord,
        }
        match cmp_i64_with_value(bag_count, func_val) {
            Ordering::Equal => {}
            ord => return ord,
        }
    }
    bag.len().cmp(&func.domain_len())
}

fn eq_bag_with_func(bag: &BagValue, func: &FuncValue) -> bool {
    if bag.len() != func.domain_len() {
        return false;
    }
    bag.entries()
        .zip(func.mapping_iter())
        .all(|((bag_key, bag_count), (func_key, func_val))| {
            bag_key == func_key && eq_i64_with_value(bag_count, func_val)
        })
}

// Helper to compare Tuple/Seq with FuncValue without allocation.
// In TLA+, tuples/sequences are functions with domain 1..n.
//
// CRITICAL (Bug #179): This MUST produce the same ordering as FuncValue::cmp
// to ensure binary search works correctly when searching for a Tuple in a set
// of Funcs. FuncValue::cmp compares entries lexicographically (like slice comparison).
//
// The tuple implicitly has entries [(1, tuple[0]), (2, tuple[1]), ...].
// We must compare these lexicographically with the func's entries, NOT by length first.
fn cmp_tuple_with_func(tuple: &[Value], func: &FuncValue) -> Ordering {
    let min_len = tuple.len().min(func.domain_len());
    for (i, (func_key, func_val)) in func.mapping_iter().take(min_len).enumerate() {
        match cmp_i64_with_value((i + 1) as i64, func_key) {
            Ordering::Equal => {}
            ord => return ord,
        }
        match tuple[i].cmp(func_val) {
            Ordering::Equal => {}
            ord => return ord,
        }
    }

    tuple.len().cmp(&func.domain_len())
}

// Helper to compare Tuple/Seq with IntIntervalFunc without allocation.
//
// CRITICAL (Bug #179): This MUST produce the same ordering as FuncValue::cmp
// to ensure binary search works correctly. Compare lexicographically, NOT by length first.
fn cmp_tuple_with_intfunc(tuple: &[Value], intfunc: &IntIntervalFunc) -> Ordering {
    let min_len = tuple.len().min(intfunc.values.len());
    for (i, (tuple_val, int_val)) in tuple
        .iter()
        .zip(intfunc.values.iter())
        .take(min_len)
        .enumerate()
    {
        let tuple_key = (i + 1) as i64;
        let intfunc_key = intfunc.min + i as i64;
        match tuple_key.cmp(&intfunc_key) {
            Ordering::Equal => {}
            ord => return ord,
        }
        match tuple_val.cmp(int_val) {
            Ordering::Equal => {}
            ord => return ord,
        }
    }

    tuple.len().cmp(&intfunc.values.len())
}

// Compare a tuple with a compact Bag's equivalent function directly. This is
// the same entry-wise ordering as `cmp_tuple_with_func`, but does not populate
// the Bag's lazily materialized Func cache.
fn cmp_tuple_with_bag(tuple: &[Value], bag: &BagValue) -> Ordering {
    for (i, (tuple_val, (bag_key, bag_count))) in tuple.iter().zip(bag.entries()).enumerate() {
        match cmp_i64_with_value((i + 1) as i64, bag_key) {
            Ordering::Equal => {}
            ord => return ord,
        }
        match cmp_i64_with_value(bag_count, tuple_val).reverse() {
            Ordering::Equal => {}
            ord => return ord,
        }
    }

    tuple.len().cmp(&bag.len())
}

// Helper to compare Record with FuncValue without allocation.
// In TLA+, records are functions with string domains.
//
// CRITICAL (Bug #179): This MUST produce the same ordering as FuncValue::cmp
// to ensure binary search works correctly. Compare lexicographically, NOT by length first.
//
// SOUNDNESS: this pairwise zip is only valid because RecordValue entries are
// stored in canonical field order (field-name string), matching the string
// order of the func's domain. When entries were NameId-sorted (interning
// order, run-dependent), this zip mismatched keys and flipped verdicts
// nondeterministically (BagsTest `h (+) h`).
fn cmp_record_with_func(record: &RecordValue, func: &FuncValue) -> Ordering {
    let mut record_iter = record.iter_str();
    let mut entries_iter = func.mapping_iter();

    loop {
        match (record_iter.next(), entries_iter.next()) {
            (Some((rec_key, rec_val)), Some((func_key, func_val))) => {
                match cmp_str_with_value(&rec_key, func_key) {
                    Ordering::Equal => {}
                    ord => return ord,
                }
                match rec_val.cmp(func_val) {
                    Ordering::Equal => {}
                    ord => return ord,
                }
            }
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (None, None) => return Ordering::Equal,
        }
    }
}

// Helper to compare FuncValue with IntIntervalFunc without allocation.
//
// CRITICAL (Bug #179): This MUST produce the same ordering as FuncValue::cmp
// to ensure binary search works correctly. Compare lexicographically, NOT by length first.
fn cmp_func_with_intfunc(func: &FuncValue, intfunc: &IntIntervalFunc) -> Ordering {
    let min_len = func.domain_len().min(intfunc.values.len());

    for (i, ((func_key, func_val), int_val)) in func
        .mapping_iter()
        .zip(intfunc.values.iter())
        .take(min_len)
        .enumerate()
    {
        match cmp_i64_with_value(intfunc.min + i as i64, func_key).reverse() {
            Ordering::Equal => {}
            ord => return ord,
        }
        match func_val.cmp(int_val) {
            Ordering::Equal => {}
            ord => return ord,
        }
    }

    func.domain_len().cmp(&intfunc.values.len())
}

// Order an integer-keyed function-like value (Tuple/Seq/IntFunc — keys 1..n or
// min..) against a Record (string keys). All of these denote functions, so the
// total order matches the entry-wise lexicographic comparison the sibling
// helpers use when routed through Func: two empties are Equal (both are the
// empty function), a shorter (empty) function precedes a non-empty one, and when
// both are non-empty the integer first-key type-orders strictly below the
// record's string first-key, so the integer-keyed value is Less. This closes the
// Eq/Ord transitivity gap where empty Tuple/Seq/IntFunc and empty Record were
// each Equal to the empty Func yet fell through to type_order (unequal) against
// one another — and it also fixes non-empty-vs-empty, which type_order had
// ordering backwards (a non-empty tuple must sort AFTER the empty record).
fn cmp_intkeyed_with_record(int_keyed_len: usize, record_len: usize) -> Ordering {
    match (int_keyed_len == 0, record_len == 0) {
        (true, true) => Ordering::Equal,
        (false, true) => Ordering::Greater,
        // empty-int-keyed vs non-empty-record (empty function precedes) and
        // both-non-empty (integer key < string key) are both Less.
        _ => Ordering::Less,
    }
}

/// Compare tuple elements with an arbitrary runtime value without wrapping
/// the elements in `Value::Tuple`.
///
/// This is exactly the ordering of `Value::Tuple(tuple.into()).cmp(rhs)`,
/// including TLA+'s alternate function representations. Keeping this beside
/// the canonical cross-type helpers is load-bearing for sorted-set binary
/// search: limiting the comparison to physical `Value::Tuple` values would
/// miss equal `Seq`, `Func`, `IntFunc`, or `Bag` elements.
pub(in crate::value) fn cmp_tuple_elements_with_value(tuple: &[Value], rhs: &Value) -> Ordering {
    match rhs {
        Value::Tuple(other) => tuple.cmp(other.as_ref()),
        Value::Seq(other) => tuple.cmp(other.flat_slice()),
        Value::Func(func) => cmp_tuple_with_func(tuple, func),
        Value::IntFunc(intfunc) => cmp_tuple_with_intfunc(tuple, intfunc),
        Value::Record(record) => cmp_intkeyed_with_record(tuple.len(), record.len()),
        Value::Bag(bag) => cmp_tuple_with_bag(tuple, bag),
        // `type_order(Value::Tuple(_)) == 4`; every same/cross-representation
        // case in that equivalence class is handled explicitly above.
        _ => 4u8.cmp(&type_order(rhs)),
    }
}

#[inline]
pub(in crate::value) fn cmp_cross_type(lhs: &Value, rhs: &Value) -> Option<Ordering> {
    match (lhs, rhs) {
        (Value::Tuple(a), Value::Func(b)) => Some(cmp_tuple_with_func(a.as_ref(), b)),
        (Value::Seq(a), Value::Func(b)) => Some(cmp_tuple_with_func(a.flat_slice(), b)),
        (Value::Func(a), Value::Tuple(b)) => Some(cmp_tuple_with_func(b.as_ref(), a).reverse()),
        (Value::Func(a), Value::Seq(b)) => Some(cmp_tuple_with_func(b.flat_slice(), a).reverse()),
        (Value::Tuple(a), Value::IntFunc(b)) => Some(cmp_tuple_with_intfunc(a.as_ref(), b)),
        (Value::Seq(a), Value::IntFunc(b)) => Some(cmp_tuple_with_intfunc(a.flat_slice(), b)),
        (Value::IntFunc(a), Value::Tuple(b)) => {
            Some(cmp_tuple_with_intfunc(b.as_ref(), a).reverse())
        }
        (Value::IntFunc(a), Value::Seq(b)) => {
            Some(cmp_tuple_with_intfunc(b.flat_slice(), a).reverse())
        }
        (Value::Tuple(a), Value::Seq(b)) => Some(a.iter().cmp(b.iter())),
        (Value::Seq(a), Value::Tuple(b)) => Some(a.iter().cmp(b.iter())),
        (Value::Record(r), Value::Func(f)) => Some(cmp_record_with_func(r, f)),
        (Value::Func(f), Value::Record(r)) => Some(cmp_record_with_func(r, f).reverse()),
        (Value::Func(f), Value::IntFunc(i)) => Some(cmp_func_with_intfunc(f, i)),
        (Value::IntFunc(i), Value::Func(f)) => Some(cmp_func_with_intfunc(f, i).reverse()),
        (Value::Tuple(a), Value::Record(r)) => {
            Some(cmp_intkeyed_with_record(a.as_ref().len(), r.len()))
        }
        (Value::Record(r), Value::Tuple(a)) => {
            Some(cmp_intkeyed_with_record(a.as_ref().len(), r.len()).reverse())
        }
        (Value::Seq(a), Value::Record(r)) => {
            Some(cmp_intkeyed_with_record(a.flat_slice().len(), r.len()))
        }
        (Value::Record(r), Value::Seq(a)) => {
            Some(cmp_intkeyed_with_record(a.flat_slice().len(), r.len()).reverse())
        }
        (Value::IntFunc(i), Value::Record(r)) => {
            Some(cmp_intkeyed_with_record(i.values.len(), r.len()))
        }
        (Value::Record(r), Value::IntFunc(i)) => {
            Some(cmp_intkeyed_with_record(i.values.len(), r.len()).reverse())
        }
        // Bag (alternate function rep): direct against Func; against the other
        // function-like reps, delegate through the bag's cached materialized
        // Func so the ordering is BY CONSTRUCTION identical to the general
        // representation's (fail-closed, rare paths).
        (Value::Bag(b), Value::Func(f)) => Some(cmp_bag_with_func(b, f)),
        (Value::Func(f), Value::Bag(b)) => Some(cmp_bag_with_func(b, f).reverse()),
        (Value::Bag(a), Value::Bag(b)) => Some(cmp_bag_with_bag(a, b)),
        (Value::Bag(bag), Value::IntFunc(i)) => {
            Some(cmp_func_with_intfunc(bag.as_func_value(), i))
        }
        (Value::IntFunc(i), Value::Bag(bag)) => {
            Some(cmp_func_with_intfunc(bag.as_func_value(), i).reverse())
        }
        (Value::Bag(bag), Value::Tuple(t)) => {
            Some(cmp_tuple_with_func(t.as_ref(), bag.as_func_value()).reverse())
        }
        (Value::Tuple(t), Value::Bag(bag)) => {
            Some(cmp_tuple_with_func(t.as_ref(), bag.as_func_value()))
        }
        (Value::Bag(bag), Value::Seq(s)) => {
            Some(cmp_tuple_with_func(s.flat_slice(), bag.as_func_value()).reverse())
        }
        (Value::Seq(s), Value::Bag(bag)) => {
            Some(cmp_tuple_with_func(s.flat_slice(), bag.as_func_value()))
        }
        (Value::Bag(bag), Value::Record(r)) => {
            Some(cmp_record_with_func(r, bag.as_func_value()).reverse())
        }
        (Value::Record(r), Value::Bag(bag)) => {
            Some(cmp_record_with_func(r, bag.as_func_value()))
        }
        _ => None,
    }
}

// Compare two compact bags. Mirrors FuncValue::cmp on the materialized forms:
// entry-wise lexicographic (elem, count), then length.
pub(super) fn cmp_bag_with_bag(a: &BagValue, b: &BagValue) -> Ordering {
    if std::ptr::eq(a, b) {
        return Ordering::Equal;
    }
    // Shared element array: only counts can differ.
    if a.elems_ptr_eq(b) {
        return a.counts().cmp(b.counts());
    }
    for ((ak, ac), (bk, bc)) in a.entries().zip(b.entries()) {
        match ak.cmp(bk) {
            Ordering::Equal => {}
            ord => return ord,
        }
        match ac.cmp(&bc) {
            Ordering::Equal => {}
            ord => return ord,
        }
    }
    a.len().cmp(&b.len())
}

pub(super) fn eq_bag_with_bag(a: &BagValue, b: &BagValue) -> bool {
    if std::ptr::eq(a, b) {
        return true;
    }
    // Different additive fingerprints => different values (the fingerprint is
    // a deterministic function of the multiset). Equal fingerprints still
    // require the deep check (collisions).
    if a.additive_fp() != b.additive_fp() {
        return false;
    }
    if a.len() != b.len() {
        return false;
    }
    if a.elems_ptr_eq(b) {
        return a.counts() == b.counts();
    }
    a.entries()
        .zip(b.entries())
        .all(|((ak, ac), (bk, bc))| ac == bc && ak == bk)
}

fn eq_tuple_with_func(tuple: &[Value], func: &FuncValue) -> bool {
    if tuple.len() != func.domain_len() {
        return false;
    }

    for (i, (func_key, func_val)) in func.mapping_iter().enumerate() {
        if !eq_i64_with_value((i + 1) as i64, func_key) {
            return false;
        }
        if &tuple[i] != func_val {
            return false;
        }
    }
    true
}

fn eq_tuple_with_intfunc(tuple: &[Value], intfunc: &IntIntervalFunc) -> bool {
    if tuple.len() != intfunc.values.len() {
        return false;
    }

    for (i, (tuple_val, int_val)) in tuple.iter().zip(intfunc.values.iter()).enumerate() {
        let tuple_key = (i + 1) as i64;
        let intfunc_key = intfunc.min + i as i64;
        if tuple_key != intfunc_key || tuple_val != int_val {
            return false;
        }
    }
    true
}

fn eq_tuple_with_bag(tuple: &[Value], bag: &BagValue) -> bool {
    tuple.len() == bag.len()
        && tuple.iter().zip(bag.entries()).enumerate().all(
            |(i, (tuple_val, (bag_key, bag_count)))| {
                eq_i64_with_value((i + 1) as i64, bag_key)
                    && eq_i64_with_value(bag_count, tuple_val)
            },
        )
}

/// Test tuple-element equality against any runtime representation without
/// constructing `Value::Tuple`.
#[inline]
pub(in crate::value) fn eq_tuple_elements_with_value(tuple: &[Value], rhs: &Value) -> bool {
    match rhs {
        Value::Tuple(other) => tuple == other.as_ref(),
        Value::Seq(other) => tuple == other.flat_slice(),
        Value::Func(func) => eq_tuple_with_func(tuple, func),
        Value::IntFunc(intfunc) => eq_tuple_with_intfunc(tuple, intfunc),
        Value::Record(record) => tuple.is_empty() && record.len() == 0,
        Value::Bag(bag) => eq_tuple_with_bag(tuple, bag),
        _ => false,
    }
}

// SOUNDNESS: pairwise zip valid only because record entries are stored in
// canonical field order (field-name string) — see cmp_record_with_func.
fn eq_record_with_func(record: &RecordValue, func: &FuncValue) -> bool {
    if record.len() != func.domain_len() {
        return false;
    }

    for ((rec_key, rec_val), (func_key, func_val)) in record.iter_str().zip(func.mapping_iter()) {
        match func_key {
            Value::String(func_key_str) if **func_key_str == *rec_key => {}
            _ => return false,
        }
        if rec_val != func_val {
            return false;
        }
    }
    true
}

fn eq_func_with_intfunc(func: &FuncValue, intfunc: &IntIntervalFunc) -> bool {
    if func.domain_len() != intfunc.values.len() {
        return false;
    }

    for (i, ((func_key, func_val), int_val)) in
        func.mapping_iter().zip(intfunc.values.iter()).enumerate()
    {
        if !eq_i64_with_value(intfunc.min + i as i64, func_key) || func_val != int_val {
            return false;
        }
    }
    true
}

#[inline]
pub(in crate::value) fn eq_cross_type(lhs: &Value, rhs: &Value) -> Option<bool> {
    match (lhs, rhs) {
        (Value::Tuple(a), Value::Func(b)) | (Value::Func(b), Value::Tuple(a)) => {
            Some(eq_tuple_with_func(a.as_ref(), b))
        }
        (Value::Seq(a), Value::Func(b)) | (Value::Func(b), Value::Seq(a)) => {
            Some(eq_tuple_with_func(a.flat_slice(), b))
        }
        (Value::Tuple(a), Value::IntFunc(b)) | (Value::IntFunc(b), Value::Tuple(a)) => {
            Some(eq_tuple_with_intfunc(a.as_ref(), b))
        }
        (Value::Seq(a), Value::IntFunc(b)) | (Value::IntFunc(b), Value::Seq(a)) => {
            Some(eq_tuple_with_intfunc(a.flat_slice(), b))
        }
        (Value::Tuple(a), Value::Seq(b)) | (Value::Seq(b), Value::Tuple(a)) => {
            Some(a.iter().eq(b.iter()))
        }
        (Value::Record(r), Value::Func(f)) | (Value::Func(f), Value::Record(r)) => {
            Some(eq_record_with_func(r, f))
        }
        (Value::Func(f), Value::IntFunc(i)) | (Value::IntFunc(i), Value::Func(f)) => {
            Some(eq_func_with_intfunc(f, i))
        }
        // An integer-keyed function-like value (Tuple/Seq/IntFunc) can equal a
        // Record only when BOTH are empty (both denote the empty function); a
        // non-empty tuple/seq/intfunc has integer keys and a non-empty record has
        // string keys, so they can never be equal. Closes the Eq transitivity gap
        // vs the empty Func (see cmp_intkeyed_with_record).
        (Value::Tuple(a), Value::Record(r)) | (Value::Record(r), Value::Tuple(a)) => {
            Some(a.as_ref().is_empty() && r.len() == 0)
        }
        (Value::Seq(a), Value::Record(r)) | (Value::Record(r), Value::Seq(a)) => {
            Some(a.flat_slice().is_empty() && r.len() == 0)
        }
        (Value::IntFunc(i), Value::Record(r)) | (Value::Record(r), Value::IntFunc(i)) => {
            Some(i.values.is_empty() && r.len() == 0)
        }
        // Bag (alternate function rep). Same structure as the cmp arms above:
        // direct against Func/Bag; delegate through the cached materialized
        // Func for the other function-like reps.
        (Value::Bag(b), Value::Func(f)) | (Value::Func(f), Value::Bag(b)) => {
            Some(eq_bag_with_func(b, f))
        }
        (Value::Bag(a), Value::Bag(b)) => Some(eq_bag_with_bag(a, b)),
        (Value::Bag(bag), Value::IntFunc(i)) | (Value::IntFunc(i), Value::Bag(bag)) => {
            Some(eq_func_with_intfunc(bag.as_func_value(), i))
        }
        (Value::Bag(bag), Value::Tuple(t)) | (Value::Tuple(t), Value::Bag(bag)) => {
            Some(eq_tuple_with_func(t.as_ref(), bag.as_func_value()))
        }
        (Value::Bag(bag), Value::Seq(s)) | (Value::Seq(s), Value::Bag(bag)) => {
            Some(eq_tuple_with_func(s.flat_slice(), bag.as_func_value()))
        }
        (Value::Bag(bag), Value::Record(r)) | (Value::Record(r), Value::Bag(bag)) => {
            Some(eq_record_with_func(r, bag.as_func_value()))
        }
        _ => None,
    }
}
