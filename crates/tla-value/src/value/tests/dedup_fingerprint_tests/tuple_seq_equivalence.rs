// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tuple/Seq fingerprint equivalence tests (Part of #3193).
//!
//! Verifies that Tuple, Seq, and IntFunc(min=1) with the same elements
//! produce identical state dedup fingerprints.

use crate::{IntIntervalFunc, Value};
use std::sync::Arc;
use crate::rp::Rp;
use super::state_value_fingerprint_unwrap;

// ---------------------------------------------------------------------------
// Tuple/Seq fingerprint equivalence tests (Part of #3193)
// ---------------------------------------------------------------------------

/// Part of #3193: Empty Tuple <<>> and empty Seq <<>> must produce the same
/// state dedup fingerprint. Before the fix, Tuple used FP64 while Seq used
/// additive, causing <<>> (Tuple) and Tail(<<x>>) (Seq) to diverge.
/// This caused unbounded state space exploration in EWD998Chan.
#[test]
fn test_tuple_seq_empty_fingerprint_equivalence() {
    // Serialize with tests that enable process-global readonly-cache mode
    // (ParallelValueInternRunGuard): cache writes are skipped while that flag
    // is set, which broke this test's additive-fp cache assertions when run
    // concurrently. All flag-flipping tests hold the same intern-state lock.
    let _intern_lock = crate::value::lock_intern_state();
    use crate::SeqValue;

    let empty: Vec<Value> = vec![];
    let tuple_val = Value::Tuple(empty.clone().into());
    let seq_val = Value::Seq(Rp::new(SeqValue::from(empty)));

    let tuple_fp = state_value_fingerprint_unwrap(&tuple_val);
    let seq_fp = state_value_fingerprint_unwrap(&seq_val);
    assert_eq!(
        tuple_fp, seq_fp,
        "empty Tuple and empty Seq must produce identical state dedup fingerprints"
    );
}

/// Part of #3193: Non-empty Tuple and Seq with the same elements must match.
/// This is the general case of the Tuple/Seq equivalence invariant.
#[test]
fn test_tuple_seq_nonempty_fingerprint_equivalence() {
    // Serialize with tests that enable process-global readonly-cache mode
    // (ParallelValueInternRunGuard): cache writes are skipped while that flag
    // is set, which broke this test's additive-fp cache assertions when run
    // concurrently. All flag-flipping tests hold the same intern-state lock.
    let _intern_lock = crate::value::lock_intern_state();
    use crate::SeqValue;

    let elems = vec![
        Value::String(Rp::from("tok")),
        Value::SmallInt(42),
        Value::Bool(true),
    ];

    let tuple_val = Value::Tuple(Rp::from(elems.as_slice()));
    let seq_val = Value::Seq(Rp::new(SeqValue::from(elems)));

    let tuple_fp = state_value_fingerprint_unwrap(&tuple_val);
    let seq_fp = state_value_fingerprint_unwrap(&seq_val);
    assert_eq!(
        tuple_fp, seq_fp,
        "Tuple and Seq with same elements must produce identical state dedup fingerprints"
    );
}

/// Part of #3193: IntFunc(min=1) must also match Tuple and Seq with same content.
/// This verifies the full three-way equivalence: Tuple == Seq == IntFunc(min=1).
#[test]
fn test_tuple_seq_intfunc_three_way_equivalence() {
    // Serialize with tests that enable process-global readonly-cache mode
    // (ParallelValueInternRunGuard): cache writes are skipped while that flag
    // is set, which broke this test's additive-fp cache assertions when run
    // concurrently. All flag-flipping tests hold the same intern-state lock.
    let _intern_lock = crate::value::lock_intern_state();
    use crate::SeqValue;

    let elems = vec![Value::SmallInt(10), Value::SmallInt(20)];

    let tuple_fp = state_value_fingerprint_unwrap(&Value::Tuple(Rp::from(elems.as_slice())));
    let seq_fp =
        state_value_fingerprint_unwrap(&Value::Seq(Rp::new(SeqValue::from(elems.clone()))));
    let intfunc_fp = state_value_fingerprint_unwrap(&Value::IntFunc(Rp::new(
        IntIntervalFunc::new(1, 2, elems),
    )));

    assert_eq!(tuple_fp, seq_fp, "Tuple and Seq must match");
    assert_eq!(seq_fp, intfunc_fp, "Seq and IntFunc(min=1) must match");
    assert_eq!(tuple_fp, intfunc_fp, "Tuple and IntFunc(min=1) must match");
}

/// Part of #3196 + #3285: Verify FP64 and additive fingerprints produce different
/// results (different algorithms). FP64 no longer cached per #3285; additive still cached.
#[test]
fn test_seq_fp64_and_additive_produce_different_results() {
    // Serialize with tests that enable process-global readonly-cache mode
    // (ParallelValueInternRunGuard): cache writes are skipped while that flag
    // is set, which broke this test's additive-fp cache assertions when run
    // concurrently. All flag-flipping tests hold the same intern-state lock.
    let _intern_lock = crate::value::lock_intern_state();
    use crate::fingerprint::value_fingerprint;
    use crate::SeqValue;

    let elems = vec![Value::SmallInt(1), Value::SmallInt(2), Value::SmallInt(3)];
    let seq = Rp::new(SeqValue::from(elems));
    let seq_val = Value::Seq(Rp::clone(&seq));

    let fp64 = value_fingerprint(&seq_val).unwrap();
    let additive = state_value_fingerprint_unwrap(&seq_val);

    // The two fingerprints must be different (different algorithms)
    assert_ne!(
        fp64, additive,
        "FP64 and additive fingerprints should differ for non-trivial sequences"
    );

    // Additive is cached; FP64 recomputed deterministically
    assert_eq!(seq.get_additive_fp(), Some(additive));
    assert_eq!(value_fingerprint(&seq_val).unwrap(), fp64);
    assert_eq!(state_value_fingerprint_unwrap(&seq_val), additive);
}

/// Part of #3196 + #3285: ordering of FP64/additive computation doesn't matter.
#[test]
fn test_seq_additive_then_fp64_ordering_independent() {
    // Serialize with tests that enable process-global readonly-cache mode
    // (ParallelValueInternRunGuard): cache writes are skipped while that flag
    // is set, which broke this test's additive-fp cache assertions when run
    // concurrently. All flag-flipping tests hold the same intern-state lock.
    let _intern_lock = crate::value::lock_intern_state();
    use crate::fingerprint::value_fingerprint;
    use crate::SeqValue;

    let elems = vec![Value::String(Rp::from("hello")), Value::Bool(true)];
    let seq = Rp::new(SeqValue::from(elems));
    let seq_val = Value::Seq(Rp::clone(&seq));

    // Compute additive first
    let additive = state_value_fingerprint_unwrap(&seq_val);
    assert_eq!(seq.get_additive_fp(), Some(additive));

    // Now compute FP64 — should not affect additive cache
    let fp64 = value_fingerprint(&seq_val).unwrap();
    assert_eq!(
        seq.get_additive_fp(),
        Some(additive),
        "additive FP should be unchanged after FP64 computation"
    );

    assert_ne!(fp64, additive);
}

/// Part of #3193: Single-element case (the exact EWD998Chan pattern).
/// inbox[i] starts as <<[type |-> "tok", ...]>> (Tuple) and after
/// PassToken the recipient gets Tail(inbox[0]) which produces Seq.
#[test]
fn test_tuple_seq_single_record_element() {
    // Serialize with tests that enable process-global readonly-cache mode
    // (ParallelValueInternRunGuard): cache writes are skipped while that flag
    // is set, which broke this test's additive-fp cache assertions when run
    // concurrently. All flag-flipping tests hold the same intern-state lock.
    let _intern_lock = crate::value::lock_intern_state();
    use crate::SeqValue;

    let record = Value::Record(crate::RecordValue::from_sorted_str_entries(vec![
        (Arc::from("color"), Value::String(Rp::from("black"))),
        (Arc::from("q"), Value::SmallInt(0)),
        (Arc::from("type"), Value::String(Rp::from("tok"))),
    ]));

    let tuple_val = Value::Tuple(Rp::from(vec![record.clone()]));
    let seq_val = Value::Seq(Rp::new(SeqValue::from(vec![record])));

    let tuple_fp = state_value_fingerprint_unwrap(&tuple_val);
    let seq_fp = state_value_fingerprint_unwrap(&seq_val);
    assert_eq!(
        tuple_fp, seq_fp,
        "single-record Tuple and Seq must produce identical fingerprints (EWD998Chan pattern)"
    );
}
