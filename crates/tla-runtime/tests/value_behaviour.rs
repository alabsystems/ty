// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Behavioural tests for the dynamically-typed [`Value`] enum and the free
//! helper functions in `types.rs`. These public APIs (the `Value` operator
//! surface, cross-type comparisons, `From` conversions, and the set-builder
//! helpers `k_subsets`/`permutations`/`seq_set`/`func_merge`/`random_element`)
//! had no dedicated coverage; this file exercises their real semantics and the
//! documented `# Panics` contracts.

use tla_runtime::{
    func_merge, is_finite_set, k_subsets, permutations, random_element, seq_set, tla_record,
    tla_set, TlaFunc, TlaRecord, TlaSet, Value,
};

// ----- Value: query methods -----

#[test]
fn value_is_true_only_for_bool_true() {
    assert!(Value::Bool(true).is_true());
    assert!(!Value::Bool(false).is_true());
    // Non-bool values are never "true" in a boolean context.
    assert!(!Value::Int(1).is_true());
    assert!(!Value::Int(0).is_true());
    assert!(!Value::Str("true".into()).is_true());
}

#[test]
fn value_not_inverts_truthiness() {
    // `Not` for Value returns a plain bool equal to !is_true().
    let not_true: bool = !Value::Bool(true);
    assert!(!not_true);
    let not_false: bool = !Value::Bool(false);
    assert!(not_false);
    // A non-bool is falsy, so its negation is true.
    let not_int: bool = !Value::Int(42);
    assert!(not_int);
}

#[test]
fn value_contains_set_membership() {
    let set = Value::Set(tla_set![Value::Int(1), Value::Int(2)]);
    assert!(set.contains(&Value::Int(1)));
    assert!(!set.contains(&Value::Int(3)));
    // Non-set values never "contain" anything.
    assert!(!Value::Int(1).contains(&Value::Int(1)));
}

#[test]
fn value_len_and_is_empty_per_variant() {
    assert_eq!(Value::Set(tla_set![Value::Int(1), Value::Int(2)]).len(), 2);
    assert_eq!(
        Value::Seq(vec![Value::Int(1), Value::Int(2), Value::Int(3)]).len(),
        3
    );
    let rec: TlaRecord<Value> =
        TlaRecord::from_fields([("a", Value::Int(1)), ("b", Value::Int(2))]);
    assert_eq!(Value::Record(rec).len(), 2);
    let f: TlaFunc<Value, Value> = [(Value::Int(1), Value::Int(9))].into_iter().collect();
    assert_eq!(Value::Func(f).len(), 1);
    // Scalars report length 0 and are "empty".
    assert_eq!(Value::Int(5).len(), 0);
    assert!(Value::Int(5).is_empty());
    assert!(Value::Set(TlaSet::<Value>::new()).is_empty());
    assert!(!Value::Seq(vec![Value::Int(0)]).is_empty());
}

#[test]
fn value_iter_yields_set_and_seq_elements_else_empty() {
    let set = Value::Set(tla_set![Value::Int(3), Value::Int(1), Value::Int(2)]);
    // Sets iterate in sorted order (BTreeSet backing).
    let collected: Vec<_> = set.iter().cloned().collect();
    assert_eq!(collected, vec![Value::Int(1), Value::Int(2), Value::Int(3)]);

    let seq = Value::Seq(vec![Value::Int(9), Value::Int(8)]);
    let collected: Vec<_> = seq.iter().cloned().collect();
    assert_eq!(collected, vec![Value::Int(9), Value::Int(8)]); // order preserved

    // A scalar yields an empty iterator.
    assert_eq!(Value::Int(7).iter().count(), 0);
}

#[test]
fn value_apply_and_get() {
    let f: TlaFunc<Value, Value> = [
        (Value::Int(1), Value::Str("a".into())),
        (Value::Int(2), Value::Str("b".into())),
    ]
    .into_iter()
    .collect();
    let fv = Value::Func(f);
    assert_eq!(fv.apply(&Value::Int(1)), Some(&Value::Str("a".into())));
    assert_eq!(fv.apply(&Value::Int(9)), None); // outside domain
    assert_eq!(Value::Int(1).apply(&Value::Int(1)), None); // not a function

    let rec = Value::Record(TlaRecord::from_fields([
        ("x", Value::Int(10)),
        ("y", Value::Int(20)),
    ]));
    assert_eq!(rec.get("x"), Some(&Value::Int(10)));
    assert_eq!(rec.get("missing"), None);
    assert_eq!(Value::Int(1).get("x"), None); // not a record
}

// ----- Value: set algebra -----

#[test]
fn value_set_algebra() {
    let a = Value::Set(tla_set![Value::Int(1), Value::Int(2), Value::Int(3)]);
    let b = Value::Set(tla_set![Value::Int(2), Value::Int(3), Value::Int(4)]);

    assert_eq!(
        a.union(&b),
        Value::Set(tla_set![
            Value::Int(1),
            Value::Int(2),
            Value::Int(3),
            Value::Int(4)
        ])
    );
    assert_eq!(
        a.intersect(&b),
        Value::Set(tla_set![Value::Int(2), Value::Int(3)])
    );
    assert_eq!(a.difference(&b), Value::Set(tla_set![Value::Int(1)]));

    let sub = Value::Set(tla_set![Value::Int(2), Value::Int(3)]);
    assert!(sub.is_subset(&a));
    assert!(!a.is_subset(&sub));
}

#[test]
fn value_set_algebra_non_set_falls_back_to_self() {
    // Documented behaviour: non-set operands return self.clone().
    let scalar = Value::Int(5);
    let set = Value::Set(tla_set![Value::Int(1)]);
    assert_eq!(scalar.union(&set), Value::Int(5));
    assert_eq!(scalar.intersect(&set), Value::Int(5));
    assert_eq!(scalar.difference(&set), Value::Int(5));
    // is_subset is false unless both are sets.
    assert!(!scalar.is_subset(&set));
    assert!(!set.is_subset(&scalar));
}

#[test]
fn value_domain_of_func() {
    let f: TlaFunc<Value, Value> = [
        (Value::Int(1), Value::Int(10)),
        (Value::Int(2), Value::Int(20)),
    ]
    .into_iter()
    .collect();
    let dom = Value::Func(f).domain();
    assert_eq!(dom, tla_set![Value::Int(1), Value::Int(2)]);
    // Non-function domain is empty.
    assert!(Value::Int(1).domain().is_empty());
}

// ----- Value: in-place / functional mutation -----

#[test]
fn value_except_returns_new_func_leaving_original() {
    let f: TlaFunc<Value, Value> = [(Value::Int(1), Value::Int(10))].into_iter().collect();
    let original = Value::Func(f);
    let updated = original.except(Value::Int(1), Value::Int(99));
    // EXCEPT is functional: original unchanged.
    assert_eq!(original.apply(&Value::Int(1)), Some(&Value::Int(10)));
    assert_eq!(updated.apply(&Value::Int(1)), Some(&Value::Int(99)));
    // EXCEPT on a new key extends the domain.
    let extended = original.except(Value::Int(2), Value::Int(20));
    assert_eq!(extended.apply(&Value::Int(2)), Some(&Value::Int(20)));
    // EXCEPT on a non-function is a no-op clone.
    assert_eq!(
        Value::Int(1).except(Value::Int(1), Value::Int(2)),
        Value::Int(1)
    );
}

#[test]
fn value_update_func_in_place() {
    let f: TlaFunc<Value, Value> = [(Value::Int(1), Value::Int(10))].into_iter().collect();
    let mut v = Value::Func(f);
    v.update(Value::Int(1), Value::Int(11));
    v.update(Value::Int(2), Value::Int(22));
    assert_eq!(v.apply(&Value::Int(1)), Some(&Value::Int(11)));
    assert_eq!(v.apply(&Value::Int(2)), Some(&Value::Int(22)));
    // update on a non-function is silently ignored.
    let mut scalar = Value::Int(0);
    scalar.update(Value::Int(1), Value::Int(2));
    assert_eq!(scalar, Value::Int(0));
}

#[test]
fn value_set_record_field_in_place() {
    let mut v = Value::Record(TlaRecord::from_fields([("a", Value::Int(1))]));
    v.set("a", Value::Int(2));
    v.set("b", Value::Int(3));
    assert_eq!(v.get("a"), Some(&Value::Int(2)));
    assert_eq!(v.get("b"), Some(&Value::Int(3)));
    // set on a non-record is ignored.
    let mut scalar = Value::Int(0);
    scalar.set("a", Value::Int(1));
    assert_eq!(scalar, Value::Int(0));
}

#[test]
fn value_insert_into_set_in_place() {
    let mut v = Value::Set(tla_set![Value::Int(1)]);
    v.insert(Value::Int(2));
    v.insert(Value::Int(1)); // duplicate ignored by set semantics
    assert_eq!(v, Value::Set(tla_set![Value::Int(1), Value::Int(2)]));
    // insert on non-set is ignored.
    let mut scalar = Value::Int(0);
    scalar.insert(Value::Int(1));
    assert_eq!(scalar, Value::Int(0));
}

#[test]
fn value_seq_push_and_first() {
    let mut v = Value::Seq(vec![Value::Int(1)]);
    v.push(Value::Int(2));
    assert_eq!(v, Value::Seq(vec![Value::Int(1), Value::Int(2)]));
    assert_eq!(v.first(), Some(&Value::Int(1)));
    // push on non-seq is ignored; first on non-seq is None.
    let mut scalar = Value::Int(0);
    scalar.push(Value::Int(9));
    assert_eq!(scalar, Value::Int(0));
    assert_eq!(Value::Int(0).first(), None);
    assert_eq!(Value::Seq(Vec::<Value>::new()).first(), None);
}

// ----- Value: arithmetic operators -----

#[test]
fn value_integer_arithmetic() {
    assert_eq!(Value::Int(6) + Value::Int(4), Value::Int(10));
    assert_eq!(Value::Int(6) - Value::Int(4), Value::Int(2));
    assert_eq!(Value::Int(6) * Value::Int(4), Value::Int(24));
    assert_eq!(Value::Int(7) / Value::Int(2), Value::Int(3));
    assert_eq!(Value::Int(7) % Value::Int(2), Value::Int(1));
    assert_eq!(-Value::Int(5), Value::Int(-5));
    assert_eq!(Value::Int(2).pow(10), Value::Int(1024));
}

#[test]
fn value_arithmetic_non_int_returns_lhs() {
    // Documented fallback: non-int operands return self.
    let s = Value::Str("x".into());
    assert_eq!(s.clone() + Value::Int(1), s);
    assert_eq!(s.clone() - Value::Int(1), s);
    assert_eq!(s.clone() * Value::Int(1), s);
    assert_eq!(-Value::Str("x".into()), Value::Str("x".into()));
    assert_eq!(Value::Str("x".into()).pow(2), Value::Str("x".into()));
}

#[test]
fn value_mixed_i64_arithmetic_both_directions() {
    // Value op i64
    assert_eq!(Value::Int(10) + 5, Value::Int(15));
    assert_eq!(Value::Int(10) - 5, Value::Int(5));
    assert_eq!(Value::Int(10) * 5, Value::Int(50));
    assert_eq!(Value::Int(10) / 5, Value::Int(2));
    assert_eq!(Value::Int(10) % 3, Value::Int(1));
    // i64 op Value
    assert_eq!(5 + Value::Int(10), Value::Int(15));
    assert_eq!(5 - Value::Int(10), Value::Int(-5));
    assert_eq!(5 * Value::Int(10), Value::Int(50));
    assert_eq!(20 / Value::Int(5), Value::Int(4));
    assert_eq!(20 % Value::Int(6), Value::Int(2));
}

// ----- Value: cross-type comparisons -----

#[test]
fn value_cross_type_equality() {
    assert_eq!(Value::Int(42), 42i64);
    assert_eq!(42i64, Value::Int(42));
    assert_ne!(Value::Int(42), 43i64);
    assert_eq!(Value::Bool(true), true);
    assert_eq!(true, Value::Bool(true));
    assert_ne!(Value::Bool(true), false);
    assert!(Value::Str("hi".into()).eq("hi"));
    assert_eq!(Value::Str("hi".into()), "hi".to_string());
    assert_eq!("hi".to_string(), Value::Str("hi".into()));
    // Mismatched variant never equals a scalar.
    assert_ne!(Value::Str("42".into()), 42i64);
}

#[test]
fn value_cross_type_equality_with_collections() {
    let s = Value::Set(tla_set![Value::Int(1), Value::Int(2)]);
    assert_eq!(s, tla_set![1i64, 2i64]);
    assert_ne!(s, tla_set![1i64, 2i64, 3i64]);

    let seq = Value::Seq(vec![Value::Int(1), Value::Int(2)]);
    assert_eq!(seq, vec![Value::Int(1), Value::Int(2)]);

    let f: TlaFunc<Value, Value> = [(Value::Int(1), Value::Int(2))].into_iter().collect();
    let fv = Value::Func(f.clone());
    assert_eq!(fv, f);

    let rec = TlaRecord::from_fields([("a", Value::Int(1))]);
    let rv = Value::Record(rec.clone());
    assert_eq!(rv, rec);
}

#[test]
fn value_cross_type_ordering() {
    use std::cmp::Ordering;
    assert_eq!(Value::Int(3).partial_cmp(&5i64), Some(Ordering::Less));
    assert_eq!(Value::Int(5).partial_cmp(&5i64), Some(Ordering::Equal));
    assert_eq!(5i64.partial_cmp(&Value::Int(3)), Some(Ordering::Greater));
    assert!(Value::Int(3) < 5i64);
    assert!(3i64 < Value::Int(5));
    // Non-int values are incomparable with i64.
    assert_eq!(Value::Str("x".into()).partial_cmp(&5i64), None);
    assert_eq!(5i64.partial_cmp(&Value::Str("x".into())), None);
}

// ----- Value: From conversions -----

#[test]
fn value_from_scalars() {
    assert_eq!(Value::from(true), Value::Bool(true));
    assert_eq!(Value::from(7i64), Value::Int(7));
    assert_eq!(Value::from("s"), Value::Str("s".into()));
    assert_eq!(Value::from(String::from("s")), Value::Str("s".into()));
}

#[test]
fn value_into_scalars_with_fallbacks() {
    // Successful coercions.
    assert_eq!(i64::from(Value::Int(9)), 9);
    assert!(bool::from(Value::Bool(true)));
    assert_eq!(String::from(Value::Str("ok".into())), "ok");
    // ModelValue extracts its name as a String.
    assert_eq!(String::from(Value::ModelValue("M".into())), "M");
    // Documented fallbacks for the wrong variant.
    assert_eq!(i64::from(Value::Bool(true)), 0);
    assert!(!bool::from(Value::Int(1)));
    assert_eq!(String::from(Value::Int(1)), "");
}

#[test]
fn value_from_collections() {
    let set: TlaSet<i64> = tla_set![1i64, 2i64];
    assert_eq!(
        Value::from(set),
        Value::Set(tla_set![Value::Int(1), Value::Int(2)])
    );

    let seq: Vec<Value> = vec![Value::Int(1)];
    assert_eq!(Value::from(seq), Value::Seq(vec![Value::Int(1)]));

    let f: TlaFunc<i64, i64> = [(1i64, 2i64)].into_iter().collect();
    assert_eq!(
        Value::from(f),
        Value::Func([(Value::Int(1), Value::Int(2))].into_iter().collect())
    );

    let rec: TlaRecord<Value> = TlaRecord::from_fields([("a", Value::Int(1))]);
    assert_eq!(Value::from(rec.clone()), Value::Record(rec));

    let pair = (Value::Int(1), Value::Int(2));
    assert_eq!(
        Value::from(pair),
        Value::Tuple(vec![Value::Int(1), Value::Int(2)])
    );
}

#[test]
fn tla_set_to_value_set() {
    let s: TlaSet<i64> = tla_set![1i64, 2i64, 3i64];
    let vs: TlaSet<Value> = s.to_value_set();
    assert_eq!(vs, tla_set![Value::Int(1), Value::Int(2), Value::Int(3)]);
}

// ----- Value: Display -----

#[test]
fn value_display_formats() {
    assert_eq!(format!("{}", Value::Bool(true)), "true");
    assert_eq!(format!("{}", Value::Int(-3)), "-3");
    // Strings are quoted via Debug formatting.
    assert_eq!(format!("{}", Value::Str("hi".into())), "\"hi\"");
    // Model values render bare.
    assert_eq!(format!("{}", Value::ModelValue("M1".into())), "M1");
}

// ----- Value: IntoIterator (owned + borrowed) -----

#[test]
fn value_into_iterator_owned() {
    let set = Value::Set(tla_set![Value::Int(2), Value::Int(1)]);
    let collected: Vec<_> = set.into_iter().collect();
    // Materialized in sorted order.
    assert_eq!(collected, vec![Value::Int(1), Value::Int(2)]);

    let tup = Value::Tuple(vec![Value::Int(9), Value::Int(8)]);
    let collected: Vec<_> = tup.into_iter().collect();
    assert_eq!(collected, vec![Value::Int(9), Value::Int(8)]);
}

#[test]
fn value_into_iterator_borrowed() {
    let seq = Value::Seq(vec![Value::Int(1), Value::Int(2)]);
    let collected: Vec<_> = (&seq).into_iter().cloned().collect();
    assert_eq!(collected, vec![Value::Int(1), Value::Int(2)]);
}

#[test]
#[should_panic(expected = "cannot iterate over")]
fn value_into_iterator_panics_on_non_iterable() {
    // Documented `# Panics`: iterating a scalar Value is a codegen bug.
    let _: Vec<_> = Value::Int(5).into_iter().collect();
}

// ----- Free helpers: k_subsets -----

#[test]
fn k_subsets_counts_and_contents() {
    let s = tla_set![1, 2, 3];
    let subsets = k_subsets(&s, 2);
    // C(3,2) = 3.
    assert_eq!(subsets.len(), 3);
    assert!(subsets.contains(&tla_set![1, 2]));
    assert!(subsets.contains(&tla_set![1, 3]));
    assert!(subsets.contains(&tla_set![2, 3]));
}

#[test]
fn k_subsets_zero_yields_only_empty_set() {
    let s = tla_set![1, 2, 3];
    let subsets = k_subsets(&s, 0);
    assert_eq!(subsets.len(), 1);
    assert!(subsets.contains(&TlaSet::<i32>::new()));
}

#[test]
fn k_subsets_full_and_oversized() {
    let s = tla_set![1, 2, 3];
    // k == n: exactly the whole set.
    let full = k_subsets(&s, 3);
    assert_eq!(full.len(), 1);
    assert!(full.contains(&tla_set![1, 2, 3]));
    // k > n: no subsets.
    assert!(k_subsets(&s, 4).is_empty());
}

// ----- Free helpers: permutations -----

#[test]
fn permutations_of_three_has_factorial_count() {
    let s = tla_set![1, 2, 3];
    let perms = permutations(&s);
    assert_eq!(perms.len(), 6); // 3!
                                // The identity permutation must be present.
    let identity: TlaFunc<i32, i32> = [(1, 1), (2, 2), (3, 3)].into_iter().collect();
    assert!(perms.contains(&identity));
    // A non-identity bijection too.
    let swap: TlaFunc<i32, i32> = [(1, 2), (2, 1), (3, 3)].into_iter().collect();
    assert!(perms.contains(&swap));
    // Every permutation maps the domain onto itself bijectively.
    for p in &perms {
        assert_eq!(p.domain(), s);
        let image: TlaSet<i32> = p.iter().map(|(_, v)| *v).collect();
        assert_eq!(image, s, "permutation must be onto");
    }
}

#[test]
fn permutations_edge_sizes() {
    // Empty set: one permutation (the empty function).
    let empty: TlaSet<i32> = TlaSet::new();
    let perms = permutations(&empty);
    assert_eq!(perms.len(), 1);
    assert!(perms.contains(&TlaFunc::<i32, i32>::new()));

    // Singleton: one permutation (identity).
    let single = tla_set![42];
    let perms = permutations(&single);
    assert_eq!(perms.len(), 1);
    let identity: TlaFunc<i32, i32> = [(42, 42)].into_iter().collect();
    assert!(perms.contains(&identity));
}

// ----- Free helpers: seq_set -----

#[test]
fn seq_set_empty_base_has_only_empty_seq() {
    let base: TlaSet<i32> = TlaSet::new();
    let seqs = seq_set(&base);
    assert_eq!(seqs.len(), 1);
    assert!(seqs.contains(&Vec::<i32>::new()));
}

#[test]
fn seq_set_includes_empty_and_bounded_sequences() {
    let base = tla_set![1, 2];
    let seqs = seq_set(&base);
    // Must always include the empty sequence.
    assert!(seqs.contains(&Vec::<i32>::new()));
    // Length-1 sequences over the base.
    assert!(seqs.contains(&vec![1]));
    assert!(seqs.contains(&vec![2]));
    // Bounded at max length 4 (n*n capped at 4 for n=2 -> 4).
    for seq in &seqs {
        assert!(seq.len() <= 4, "seq_set must bound sequence length");
        for elem in seq {
            assert!(base.contains(elem), "elements must come from base set");
        }
    }
    // A length-4 sequence is reachable at this bound.
    assert!(seqs.contains(&vec![1, 1, 1, 1]));
}

// ----- Free helpers: func_merge -----

#[test]
fn func_merge_left_takes_priority() {
    let f: TlaFunc<i32, &str> = [(1, "f1"), (2, "f2")].into_iter().collect();
    let g: TlaFunc<i32, &str> = [(2, "g2"), (3, "g3")].into_iter().collect();
    let merged = func_merge(&f, &g);
    // Domain is the union.
    assert_eq!(merged.domain(), tla_set![1, 2, 3]);
    // f wins on the shared key 2.
    assert_eq!(merged.apply(&2), Some(&"f2"));
    // g-only key preserved.
    assert_eq!(merged.apply(&3), Some(&"g3"));
    // f-only key preserved.
    assert_eq!(merged.apply(&1), Some(&"f1"));
}

#[test]
fn func_merge_with_empty_operands() {
    let f: TlaFunc<i32, i32> = [(1, 10)].into_iter().collect();
    let empty: TlaFunc<i32, i32> = TlaFunc::new();
    assert_eq!(func_merge(&f, &empty), f);
    assert_eq!(func_merge(&empty, &f), f);
    assert_eq!(func_merge(&empty, &empty), TlaFunc::<i32, i32>::new());
}

// ----- Free helpers: random_element -----

#[test]
fn random_element_is_deterministic_minimum() {
    // Despite the name, returns the smallest element under sort order.
    let s = tla_set![5, 1, 3, 2, 4];
    assert_eq!(random_element(&s), 1);
    // Repeated calls are stable.
    assert_eq!(random_element(&s), random_element(&s));
    let strs = tla_set!["banana", "apple", "cherry"];
    assert_eq!(random_element(&strs), "apple");
}

#[test]
#[should_panic(expected = "RandomElement requires non-empty set")]
fn random_element_panics_on_empty_set() {
    // Documented `# Panics`: CHOOSE/RandomElement undefined on the empty set.
    let empty: TlaSet<i32> = TlaSet::new();
    let _ = random_element(&empty);
}

// ----- Free helpers: is_finite_set -----

#[test]
fn is_finite_set_always_true() {
    // In-memory sets are always finite.
    assert!(is_finite_set(tla_set![1, 2, 3]));
    assert!(is_finite_set(TlaSet::<i32>::new()));
}

// ----- macro: tla_record! / tla_set! empty forms -----

#[test]
fn macros_empty_constructors() {
    let r: TlaRecord<i32> = tla_record![];
    assert!(r.fields().is_empty());
    let s: TlaSet<i32> = tla_set![];
    assert!(s.is_empty());
    // Non-empty record macro builds the expected fields.
    let r2 = tla_record![a => 1, b => 2];
    assert_eq!(r2.get("a"), Some(&1));
    assert_eq!(r2.get("b"), Some(&2));
}
