// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_payload_witness_compact_values_match_array_state_without_fp_cache() {
    let registry = VarRegistry::from_names(["x", "y"]);
    let mut state = ArrayState::from_values(vec![Value::int(1), Value::Bool(true)]);
    let fp = state.fingerprint(&registry);
    state.set_cached_fingerprint(fp);

    let witness = StatePayloadWitness::from_array_state(&state);
    assert_eq!(witness.kind(), StatePayloadWitnessKind::CompactValueSlots);
    assert_eq!(
        witness.payload_bytes(),
        2 * std::mem::size_of::<tla_value::CompactValue>()
    );
    assert!(witness.matches_array_state(&state));

    let different = ArrayState::from_values(vec![Value::int(1), Value::Bool(false)]);
    assert!(!witness.matches_array_state(&different));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_payload_witness_flat_i64_slots_are_typed() {
    let witness = StatePayloadWitness::from_flat_i64_slots(&[1, 0, 7]);
    assert_eq!(witness.kind(), StatePayloadWitnessKind::FlatI64Slots);
    assert_eq!(witness.payload_bytes(), 3 * std::mem::size_of::<i64>());
    assert!(witness.matches_flat_i64_slots(&[1, 0, 7]));
    assert!(!witness.matches_flat_i64_slots(&[1, 0, 8]));

    let same_numbers_as_tla_values =
        ArrayState::from_values(vec![Value::int(1), Value::int(0), Value::int(7)]);
    assert!(!witness.matches_array_state(&same_numbers_as_tla_values));
}
