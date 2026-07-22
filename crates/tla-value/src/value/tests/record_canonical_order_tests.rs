// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Canonical record field order: field-name STRING, never NameId interning order.
//!
//! Regression tests for the BagsTest `h (+) h` nondeterministic-verdict bug:
//! `RecordValue` entries were sorted by NameId (interning order — run-dependent),
//! while string-keyed `FuncValue`s sort by string content. Cross-type eq/cmp
//! zipped the two orders pairwise, so `record = func` flipped with the per-run
//! interning order whenever a field set wasn't interned alphabetically.
//!
//! All tests intern the LATER-alphabetical name FIRST to force
//! NameId-order != string-order (the order that used to break).

use crate::value::{RecordBuilder, RecordValue, Value};
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use crate::rp::Rp;
use tla_core::intern_name;

/// Intern `zz` before `aa` so NameId numeric order REVERSES string order.
/// Unique prefixes keep these names out of other tests' interning.
fn reverse_interned_pair(tag: &str) -> (String, String) {
    let zz = format!("zzcanon_{tag}");
    let aa = format!("aacanon_{tag}");
    let zz_id = intern_name(&zz);
    let aa_id = intern_name(&aa);
    // The premise of these tests: zz interned first => smaller NameId.
    assert!(zz_id.0 < aa_id.0, "test setup: zz must be interned first");
    (zz, aa)
}

fn record_zz_aa(zz: &str, aa: &str, zz_val: i64, aa_val: i64) -> RecordValue {
    let mut b = RecordBuilder::new();
    // Insert in NameId order (zz first) — build() must sort by STRING.
    b.insert_str(zz, Value::SmallInt(zz_val));
    b.insert_str(aa, Value::SmallInt(aa_val));
    b.build()
}

#[test]
fn record_entries_sorted_by_field_string_not_nameid() {
    let (zz, aa) = reverse_interned_pair("entries");
    let rec = record_zz_aa(&zz, &aa, 1, 2);
    let keys: Vec<Arc<str>> = rec.key_strings().collect();
    assert_eq!(keys.len(), 2);
    assert_eq!(&*keys[0], aa.as_str(), "iteration must be string-sorted");
    assert_eq!(&*keys[1], zz.as_str());
}

#[test]
fn record_eq_string_keyed_func_with_reverse_intern_order() {
    let (zz, aa) = reverse_interned_pair("eq");
    let rec = Value::Record(record_zz_aa(&zz, &aa, 1, 2));
    // The equivalent function, as produced by e.g. Bags (+): string-sorted.
    let func = rec
        .to_func_coerced()
        .map(|f| Value::Func(Rp::new(f)))
        .expect("record coerces to func");
    assert_eq!(rec, func, "record must equal its function form");
    assert_eq!(func, rec, "symmetric");
    assert_eq!(
        rec.cmp(&func),
        Ordering::Equal,
        "cmp must agree with eq across Record/Func"
    );
}

#[test]
fn record_func_fingerprint_parity_with_reverse_intern_order() {
    let (zz, aa) = reverse_interned_pair("fp");
    let rec = Value::Record(record_zz_aa(&zz, &aa, 1, 2));
    let func = rec
        .to_func_coerced()
        .map(|f| Value::Func(Rp::new(f)))
        .expect("record coerces to func");
    let rec_fp = rec.fingerprint_extend(0).expect("record fp");
    let func_fp = func.fingerprint_extend(0).expect("func fp");
    assert_eq!(
        rec_fp, func_fp,
        "semantically equal Record/Func must fingerprint identically (state dedup)"
    );
}

#[test]
fn record_func_hash_parity_with_reverse_intern_order() {
    let (zz, aa) = reverse_interned_pair("hash");
    let rec = Value::Record(record_zz_aa(&zz, &aa, 1, 2));
    let func = rec
        .to_func_coerced()
        .map(|f| Value::Func(Rp::new(f)))
        .expect("record coerces to func");
    let mut h1 = DefaultHasher::new();
    rec.hash(&mut h1);
    let mut h2 = DefaultHasher::new();
    func.hash(&mut h2);
    assert_eq!(
        h1.finish(),
        h2.finish(),
        "eq values must hash identically (HashMap contract)"
    );
}

#[test]
fn record_record_cmp_uses_string_order() {
    let (zz, aa) = reverse_interned_pair("cmp");
    // r_aa = [aa |-> 1], r_zz = [zz |-> 0]. String order: "aa..." < "zz...",
    // NameId order: zz < aa (reversed). The total order must use strings so
    // mixed Record/Func sorted sets stay transitive.
    let mut b = RecordBuilder::new();
    b.insert_str(&aa, Value::SmallInt(1));
    let r_aa = Value::Record(b.build());
    let mut b = RecordBuilder::new();
    b.insert_str(&zz, Value::SmallInt(0));
    let r_zz = Value::Record(b.build());
    assert_eq!(r_aa.cmp(&r_zz), Ordering::Less);
    assert_eq!(r_zz.cmp(&r_aa), Ordering::Greater);

    // Cross-check transitivity against the func forms.
    let f_zz = r_zz
        .to_func_coerced()
        .map(|f| Value::Func(Rp::new(f)))
        .expect("coerce");
    assert_eq!(r_aa.cmp(&f_zz), Ordering::Less, "record vs func consistent");
    assert_eq!(f_zz.cmp(&r_aa), Ordering::Greater);
}

#[test]
fn record_field_lookup_survives_string_sorting() {
    // >8 fields exercises the binary-search lookup path, which must search
    // by the canonical string order. Insert in reverse-alphabetical order
    // with reverse-alphabetical interning.
    let names: Vec<String> = (0..12)
        .rev()
        .map(|i| format!("f{i:02}_canon_lookup"))
        .collect();
    let mut b = RecordBuilder::new();
    for (val, name) in names.iter().enumerate() {
        b.insert_str(name, Value::SmallInt(val as i64));
    }
    let rec = b.build();
    assert_eq!(rec.len(), 12);
    for (val, name) in names.iter().enumerate() {
        assert_eq!(
            rec.get(name),
            Some(&Value::SmallInt(val as i64)),
            "lookup of {name} after canonical sort"
        );
    }
    // EXCEPT-style update on the >8-field record (binary-search mutation path).
    let updated = rec.insert(intern_name(&names[0]), Value::SmallInt(99));
    assert_eq!(updated.get(&names[0]), Some(&Value::SmallInt(99)));
    assert_eq!(updated.len(), 12);
    // Insert a NEW field that sorts to the front by string.
    let new_name = "a00_canon_lookup";
    let grown = rec.insert(intern_name(new_name), Value::SmallInt(-1));
    assert_eq!(grown.len(), 13);
    assert_eq!(grown.get(new_name), Some(&Value::SmallInt(-1)));
    let keys: Vec<Arc<str>> = grown.key_strings().collect();
    assert_eq!(
        &*keys[0], new_name,
        "new field must sort to string position"
    );
}

#[test]
fn bag_cup_style_record_func_roundtrip() {
    // Mirror of the BagsTest `h (+) h = [...]` ASSUME with hostile interning:
    // (+) coerces both sides to funcs, sums counts, and returns a Func; the
    // spec compares it against a Record literal.
    let (zz, aa) = reverse_interned_pair("bagcup");
    let h = Value::Record(record_zz_aa(&zz, &aa, 3, 2));
    let func = h.to_func_coerced().expect("coerce");
    let doubled: Vec<(Value, Value)> = func
        .mapping_iter()
        .map(|(k, v)| {
            let n = match v {
                Value::SmallInt(n) => *n,
                _ => panic!("int count"),
            };
            (k.clone(), Value::SmallInt(n * 2))
        })
        .collect();
    let cup = Value::Func(Rp::new(crate::value::FuncValue::from_sorted_entries(
        doubled,
    )));
    let expected = Value::Record(record_zz_aa(&zz, &aa, 6, 4));
    assert_eq!(cup, expected, "h (+) h = [zz |-> 6, aa |-> 4] must hold");
}
