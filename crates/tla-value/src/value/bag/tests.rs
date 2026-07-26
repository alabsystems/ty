// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Representation-equivalence tests: a compact `Value::Bag` must be
//! observationally IDENTICAL to the equivalent general `Value::Func` —
//! eq/cmp/hash, state-dedup fingerprint, TLC FP64 fingerprint, and
//! DOMAIN/iteration order. Any divergence is a state-dedup soundness bug.

use super::super::{FuncValue, Value};
use super::BagValue;
use crate::dedup_fingerprint::state_value_fingerprint;
use crate::fingerprint::FP64_INIT;
use crate::rp::Rp;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
fn std_hash(v: &Value) -> u64 {
    let mut h = DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

/// Simple deterministic LCG for reproducible "random" bags.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn record(fields: Vec<(&str, Value)>) -> Value {
    let mut entries: Vec<(tla_core::NameId, Value)> = fields
        .into_iter()
        .map(|(k, v)| (tla_core::intern_name(k), v))
        .collect();
    entries.sort_by(|a, b| {
        tla_core::resolve_name_id(a.0)
            .as_ref()
            .cmp(tla_core::resolve_name_id(b.0).as_ref())
    });
    Value::Record(crate::RecordValue::from_sorted_entries(entries))
}

/// EWD998-shaped message records plus scalar oddballs.
fn random_elem(rng: &mut Lcg) -> Value {
    match rng.below(5) {
        0 => record(vec![("type", Value::string("pl"))]),
        1 => record(vec![
            ("type", Value::string("tok")),
            ("q", Value::SmallInt(rng.below(7) as i64 - 3)),
            (
                "color",
                Value::string(if rng.below(2) == 0 { "black" } else { "white" }),
            ),
        ]),
        2 => Value::SmallInt(rng.below(100) as i64),
        3 => Value::string(["a", "b", "c", "zz"][rng.below(4) as usize]),
        _ => Value::Tuple(Rp::from(vec![
            Value::SmallInt(rng.below(5) as i64),
            Value::string("x"),
        ])),
    }
}

/// Random sorted, unique (elem, SmallInt(count)) entries.
fn random_entries(rng: &mut Lcg, max_len: u64) -> Vec<(Value, Value)> {
    let mut entries: Vec<(Value, Value)> = Vec::new();
    for _ in 0..rng.below(max_len + 1) {
        let e = random_elem(rng);
        if entries.iter().any(|(k, _)| *k == e) {
            continue;
        }
        let c = rng.below(4) as i64 + 1;
        entries.push((e, Value::SmallInt(c)));
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

fn make_pair(entries: Vec<(Value, Value)>) -> (Value, Value) {
    let bag = BagValue::try_from_entries(entries.clone())
        .expect("eligible entries must build a compact bag");
    let func = FuncValue::from_sorted_entries(entries);
    (Value::Bag(Rp::new(bag)), Value::Func(Rp::new(func)))
}

#[test]
fn rep_equivalence_random_bags() {
    let mut rng = Lcg(0x5eed_cafe);
    for round in 0..500 {
        let entries = random_entries(&mut rng, 6);
        let (bag, func) = make_pair(entries);

        // Eq + Ord, both directions.
        assert_eq!(bag, func, "round {round}: bag != equivalent func");
        assert_eq!(func, bag, "round {round}: func != equivalent bag");
        assert_eq!(
            bag.cmp(&func),
            std::cmp::Ordering::Equal,
            "round {round}: cmp(bag, func) != Equal"
        );
        assert_eq!(
            func.cmp(&bag),
            std::cmp::Ordering::Equal,
            "round {round}: cmp(func, bag) != Equal"
        );

        // Rust Hash (Hash/Eq contract).
        assert_eq!(std_hash(&bag), std_hash(&func), "round {round}: Hash");

        // State-dedup (additive) fingerprint — THE dedup soundness invariant.
        assert_eq!(
            state_value_fingerprint(&bag).unwrap(),
            state_value_fingerprint(&func).unwrap(),
            "round {round}: state dedup fingerprint"
        );

        // TLC FP64 fingerprint. Mixed-type domains error in TLC normalization
        // (string vs int is not comparable in TLC) — the bag must produce the
        // SAME outcome as the func (both Ok-and-equal or both the same error).
        let bag_fp64 = bag.fingerprint_extend(FP64_INIT);
        let func_fp64 = func.fingerprint_extend(FP64_INIT);
        match (bag_fp64, func_fp64) {
            (Ok(a), Ok(b)) => assert_eq!(a, b, "round {round}: FP64 fingerprint"),
            (Err(ea), Err(eb)) => assert_eq!(
                ea.to_string(),
                eb.to_string(),
                "round {round}: FP64 error mismatch"
            ),
            (a, b) => panic!("round {round}: FP64 outcome diverged: {a:?} vs {b:?}"),
        }

        // Debug/Display rendering (traces must be rep-independent).
        assert_eq!(
            format!("{bag:?}"),
            format!("{func:?}"),
            "round {round}: Debug"
        );
        assert_eq!(
            format!("{bag}"),
            format!("{func}"),
            "round {round}: Display"
        );
    }
}

#[test]
fn domain_iteration_and_choose_order_matches_func() {
    let mut rng = Lcg(0xd07a11);
    for _ in 0..200 {
        let entries = random_entries(&mut rng, 6);
        let func = FuncValue::from_sorted_entries(entries.clone());
        let Ok(bag) = BagValue::try_from_entries(entries) else {
            panic!("eligible entries must build a compact bag");
        };
        // DOMAIN iteration order (and hence CHOOSE over DOMAIN) must be
        // identical to the general representation's.
        let bag_domain: Vec<Value> = match bag.domain_set_value() {
            Value::Set(ref s) => s.iter().cloned().collect(),
            ref other => panic!("DOMAIN bag must be a Set, got {other:?}"),
        };
        let func_domain: Vec<Value> = func.domain_iter().cloned().collect();
        assert_eq!(bag_domain, func_domain, "DOMAIN order mismatch");
        // Entry iteration order likewise.
        let bag_entries: Vec<(Value, Value)> = bag
            .entries()
            .map(|(e, c)| (e.clone(), Value::SmallInt(c)))
            .collect();
        let func_entries: Vec<(Value, Value)> = func
            .mapping_iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        assert_eq!(bag_entries, func_entries, "entry iteration order mismatch");
    }
}

/// Reference general-path BagAdd (mirrors builtin_bagsext).
fn general_bag_add(entries: &[(Value, Value)], e: &Value) -> Vec<(Value, Value)> {
    let mut out: Vec<(Value, Value)> = entries.to_vec();
    if let Some(entry) = out.iter_mut().find(|(k, _)| k == e) {
        let c = entry.1.as_i64().unwrap();
        entry.1 = Value::SmallInt(c + 1);
    } else {
        out.push((e.clone(), Value::SmallInt(1)));
        out.sort_by(|a, b| a.0.cmp(&b.0));
    }
    out
}

/// Reference general-path BagRemove.
fn general_bag_remove(entries: &[(Value, Value)], e: &Value) -> Vec<(Value, Value)> {
    let mut out: Vec<(Value, Value)> = Vec::new();
    for (k, v) in entries {
        if k == e {
            let c = v.as_i64().unwrap() - 1;
            if c > 0 {
                out.push((k.clone(), Value::SmallInt(c)));
            }
        } else {
            out.push((k.clone(), v.clone()));
        }
    }
    out
}

#[test]
fn bag_ops_match_general_semantics_and_fingerprints() {
    let mut rng = Lcg(0x0b5e55ed);
    for round in 0..300 {
        let entries = random_entries(&mut rng, 5);
        let bag = BagValue::try_from_entries(entries.clone()).unwrap();
        // Probe with both present and fresh elements.
        let probe = if !entries.is_empty() && rng.below(2) == 0 {
            entries[rng.below(entries.len() as u64) as usize].0.clone()
        } else {
            random_elem(&mut rng)
        };

        // BagAdd
        let expected_add = general_bag_add(&entries, &probe);
        let added = bag.bag_add(&probe).expect("bag_add must succeed here");
        let expected_func = FuncValue::from_sorted_entries(expected_add.clone());
        let added_val = Value::Bag(Rp::new(added));
        let expected_val = Value::Func(Rp::new(expected_func));
        assert_eq!(added_val, expected_val, "round {round}: BagAdd value");
        assert_eq!(
            state_value_fingerprint(&added_val).unwrap(),
            state_value_fingerprint(&expected_val).unwrap(),
            "round {round}: BagAdd incremental fingerprint"
        );

        // BagRemove
        let expected_rem = general_bag_remove(&entries, &probe);
        let removed_val = match bag.bag_remove(&probe) {
            Some(b) => Value::Bag(Rp::new(b)),
            None => {
                // Absent element: general path rebuilds identical entries.
                assert_eq!(expected_rem, entries, "round {round}: absent remove");
                continue;
            }
        };
        let expected_rem_val = Value::Func(Rp::new(FuncValue::from_sorted_entries(expected_rem)));
        assert_eq!(removed_val, expected_rem_val, "round {round}: BagRemove");
        assert_eq!(
            state_value_fingerprint(&removed_val).unwrap(),
            state_value_fingerprint(&expected_rem_val).unwrap(),
            "round {round}: BagRemove incremental fingerprint"
        );
    }
}

#[test]
fn bag_cup_and_diff_match_general_semantics() {
    let mut rng = Lcg(0xc0ffee);
    for round in 0..200 {
        let e1 = random_entries(&mut rng, 5);
        let e2 = random_entries(&mut rng, 5);
        let b1 = BagValue::try_from_entries(e1.clone()).unwrap();
        let b2 = BagValue::try_from_entries(e2.clone()).unwrap();

        // Reference cup
        let mut cup: Vec<(Value, Value)> = e1.clone();
        for (k, v) in &e2 {
            if let Some(entry) = cup.iter_mut().find(|(ck, _)| ck == k) {
                entry.1 = Value::SmallInt(entry.1.as_i64().unwrap() + v.as_i64().unwrap());
            } else {
                cup.push((k.clone(), v.clone()));
            }
        }
        cup.sort_by(|a, b| a.0.cmp(&b.0));
        let cup_bag = Value::Bag(Rp::new(b1.bag_cup(&b2).unwrap()));
        let cup_func = Value::Func(Rp::new(FuncValue::from_sorted_entries(cup)));
        assert_eq!(cup_bag, cup_func, "round {round}: BagCup");
        assert_eq!(
            state_value_fingerprint(&cup_bag).unwrap(),
            state_value_fingerprint(&cup_func).unwrap(),
            "round {round}: BagCup fingerprint"
        );

        // Reference diff
        let mut diff: Vec<(Value, Value)> = Vec::new();
        for (k, v) in &e1 {
            let sub = e2
                .iter()
                .find(|(k2, _)| k2 == k)
                .map_or(0, |(_, v2)| v2.as_i64().unwrap());
            let c = v.as_i64().unwrap() - sub;
            if c > 0 {
                diff.push((k.clone(), Value::SmallInt(c)));
            }
        }
        let diff_bag = Value::Bag(Rp::new(b1.bag_diff(&b2)));
        let diff_func = Value::Func(Rp::new(FuncValue::from_sorted_entries(diff)));
        assert_eq!(diff_bag, diff_func, "round {round}: BagDiff");
        assert_eq!(
            state_value_fingerprint(&diff_bag).unwrap(),
            state_value_fingerprint(&diff_func).unwrap(),
            "round {round}: BagDiff fingerprint"
        );
    }
}

#[test]
fn fail_closed_ineligible_entries() {
    // Zero count → ineligible.
    let zero = vec![(Value::SmallInt(1), Value::SmallInt(0))];
    assert!(BagValue::try_from_entries(zero).is_err());
    // Negative count → ineligible.
    let neg = vec![(Value::SmallInt(1), Value::SmallInt(-2))];
    assert!(BagValue::try_from_entries(neg).is_err());
    // Non-integer count → ineligible.
    let non_int = vec![(Value::SmallInt(1), Value::string("x"))];
    assert!(BagValue::try_from_entries(non_int).is_err());
    // BigInt count beyond i64 → ineligible.
    let huge = num_bigint::BigInt::from(i64::MAX) + 10;
    let big = vec![(Value::SmallInt(1), Value::Int(Rp::new(huge)))];
    assert!(BagValue::try_from_entries(big).is_err());
    // try_from_func mirrors the same checks.
    let f = FuncValue::from_sorted_entries(vec![(Value::SmallInt(1), Value::SmallInt(0))]);
    assert!(BagValue::try_from_func(&f).is_none());
}

#[test]
fn bag_add_count_overflow_fails_closed() {
    let entries = vec![(Value::SmallInt(7), Value::SmallInt(i64::MAX))];
    let bag = BagValue::try_from_entries(entries).unwrap();
    assert!(bag.bag_add(&Value::SmallInt(7)).is_none());
    // A different element still works.
    assert!(bag.bag_add(&Value::SmallInt(8)).is_some());
}

#[test]
fn empty_bag_equals_all_empty_function_reps() {
    let bag = Value::Bag(BagValue::empty_arc());
    let func = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![])));
    let tuple = Value::Tuple(Rp::from(Vec::<Value>::new()));
    let seq = Value::seq(Vec::<Value>::new());
    let rec = Value::Record(crate::RecordValue::from_sorted_entries(vec![]));
    for other in [&func, &tuple, &seq, &rec] {
        assert_eq!(&bag, other, "empty bag == empty {other:?}");
        assert_eq!(other, &bag);
        assert_eq!(bag.cmp(other), std::cmp::Ordering::Equal);
        assert_eq!(other.cmp(&bag), std::cmp::Ordering::Equal);
    }
    assert_eq!(std_hash(&bag), std_hash(&func));
    assert_eq!(
        state_value_fingerprint(&bag).unwrap(),
        state_value_fingerprint(&func).unwrap()
    );
}

#[test]
fn cross_type_bag_vs_tuple_seq_intfunc() {
    // Bag {1 ↦ 1, 2 ↦ 2} == tuple <<1, 2>> == the same Func.
    let entries = vec![
        (Value::SmallInt(1), Value::SmallInt(1)),
        (Value::SmallInt(2), Value::SmallInt(2)),
    ];
    let bag = Value::Bag(Rp::new(BagValue::try_from_entries(entries).unwrap()));
    let tuple = Value::Tuple(Rp::from(vec![Value::SmallInt(1), Value::SmallInt(2)]));
    let seq = Value::seq(vec![Value::SmallInt(1), Value::SmallInt(2)]);
    assert_eq!(bag, tuple);
    assert_eq!(tuple, bag);
    assert_eq!(bag, seq);
    assert_eq!(seq, bag);
    assert_eq!(bag.cmp(&tuple), std::cmp::Ordering::Equal);
    assert_eq!(tuple.cmp(&bag), std::cmp::Ordering::Equal);
    // And a non-equal tuple orders identically against bag and func.
    let other = Value::Tuple(Rp::from(vec![Value::SmallInt(1), Value::SmallInt(3)]));
    let func = bag
        .to_func_coerced()
        .map(|f| Value::Func(Rp::new(f)))
        .unwrap();
    assert_eq!(bag.cmp(&other), func.cmp(&other));
    assert_eq!(other.cmp(&bag), other.cmp(&func));
}

#[test]
fn sorted_set_ordering_is_rep_independent() {
    // Sets containing a bag must sort exactly as with the equivalent func —
    // CHOOSE over such sets is then rep-independent.
    let mut rng = Lcg(0xab5ac7);
    for _ in 0..100 {
        let entries = random_entries(&mut rng, 4);
        let (bag, func) = make_pair(entries);
        let mut others: Vec<Value> = (0..5)
            .map(|_| {
                let e = random_entries(&mut rng, 3);
                Value::Func(Rp::new(FuncValue::from_sorted_entries(e)))
            })
            .collect();
        others.push(Value::SmallInt(3));
        others.push(Value::string("s"));

        let mut with_bag = others.clone();
        with_bag.push(bag);
        let mut with_func = others;
        with_func.push(func.clone());
        with_bag.sort();
        with_func.sort();
        for (a, b) in with_bag.iter().zip(with_func.iter()) {
            assert_eq!(a, b, "sorted order diverged between representations");
        }
    }
}

#[test]
fn apply_and_domain_contains() {
    let entries = vec![
        (Value::SmallInt(2), Value::SmallInt(3)),
        (Value::string("m"), Value::SmallInt(1)),
    ];
    let entries = {
        let mut e = entries;
        e.sort_by(|a, b| a.0.cmp(&b.0));
        e
    };
    let bag = BagValue::try_from_entries(entries).unwrap();
    assert_eq!(bag.apply(&Value::SmallInt(2)), Some(Value::SmallInt(3)));
    assert_eq!(bag.apply(&Value::string("m")), Some(Value::SmallInt(1)));
    assert_eq!(bag.apply(&Value::SmallInt(9)), None);
    assert_eq!(bag.count_of(&Value::SmallInt(9)), 0);
    assert!(bag.domain_contains(&Value::string("m")));
    assert_eq!(bag.cardinality(), Some(4));
}
