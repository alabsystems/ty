// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Storage-mode unit tests.
//!
//! Part of #3436: extracted from `storage_modes.rs`.

use super::*;
use crate::arena::BulkStateStorage;
use crate::config::Config;
use crate::storage::{
    BatchInsertedIndexAdmission, CapacityStatus, InsertOutcome, LookupOutcome, StorageFault,
};
use crate::test_support::parse_module;
use crate::Value;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn minimal_module() -> tla_core::ast::Module {
    parse_module(
        r#"
---- MODULE StorageModeTest ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
    )
}

fn minimal_config() -> Config {
    Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    }
}

fn view_module() -> tla_core::ast::Module {
    parse_module(
        r#"
---- MODULE StorageModeViewTest ----
VARIABLES x, y
Init == /\ x = 0 /\ y = 0
Next == /\ x' = x /\ y' = y + 1
View == x
====
"#,
    )
}

fn view_config() -> Config {
    Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        view: Some("View".to_string()),
        ..Default::default()
    }
}

/// Fingerprint set that admits the first member of a prepared batch and then
/// reports a storage fault. This exercises prefix preservation at the exact
/// boundary where a bulk backend fails.
struct FaultAfterFirstInsertSet {
    len: AtomicUsize,
}

impl FaultAfterFirstInsertSet {
    fn new() -> Self {
        Self {
            len: AtomicUsize::new(0),
        }
    }
}

impl tla_mc_core::FingerprintSet<Fingerprint> for FaultAfterFirstInsertSet {
    fn insert_checked(&self, _fingerprint: Fingerprint) -> InsertOutcome {
        InsertOutcome::StorageFault(StorageFault::new("test", "insert", "injected fault"))
    }

    fn contains_checked(&self, _fingerprint: Fingerprint) -> LookupOutcome {
        LookupOutcome::Absent
    }

    fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    fn has_errors(&self) -> bool {
        true
    }

    fn dropped_count(&self) -> usize {
        1
    }

    fn capacity_status(&self) -> CapacityStatus {
        CapacityStatus::Normal
    }
}

impl crate::storage::FingerprintSet for FaultAfterFirstInsertSet {
    fn insert_prechecked_absent_batch_inserted_indices_checked_into(
        &self,
        fingerprints: &[Fingerprint],
        admission: &mut BatchInsertedIndexAdmission,
    ) {
        admission.clear();
        admission.inserted_indices.reserve(fingerprints.len());
        if fingerprints.is_empty() {
            return;
        }

        admission.attempted = 1;
        admission.inserted_indices.push(0);
        self.len.fetch_add(1, Ordering::Relaxed);

        if fingerprints.len() > 1 {
            admission.attempted += 1;
            admission.fault = Some(StorageFault::new("test", "insert", "injected fault"));
        }
    }
}

// -----------------------------------------------------------------------
// FullStateStorage tests
// -----------------------------------------------------------------------

#[test]
fn full_state_dequeue_present_returns_state_and_removes_from_seen() {
    let module = minimal_module();
    let config = minimal_config();
    let mut mc = ModelChecker::new(&module, &config);
    mc.set_store_states(true);

    let fp = Fingerprint(42);
    let state = ArrayState::from_values(vec![Value::int(7)]);
    mc.state_storage.seen.insert(fp, state);

    let mut storage = FullStateStorage;
    let result = storage.dequeue((fp, 5), &mut mc).unwrap().unwrap();
    assert_eq!(result.0, fp);
    assert_eq!(result.2, 5);
    // State should be removed from seen during dequeue
    assert!(!mc.state_storage.seen.contains_key(&fp));
}

#[test]
fn full_state_dequeue_missing_fp_returns_none_and_increments_phantom() {
    let module = minimal_module();
    let config = minimal_config();
    let mut mc = ModelChecker::new(&module, &config);

    let initial_phantoms = mc.stats.phantom_dequeues;
    let mut storage = FullStateStorage;
    let result = storage.dequeue((Fingerprint(999), 0), &mut mc).unwrap();
    assert!(result.is_none());
    assert_eq!(mc.stats.phantom_dequeues, initial_phantoms + 1);
}

#[test]
fn full_state_return_current_reinserts_into_seen() {
    let module = minimal_module();
    let config = minimal_config();
    let mut mc = ModelChecker::new(&module, &config);
    mc.set_store_states(true);

    let fp = Fingerprint(42);
    let state = ArrayState::from_values(vec![Value::int(7)]);
    assert!(!mc.state_storage.seen.contains_key(&fp));

    let mut storage = FullStateStorage;
    storage.return_current(fp, state, &mut mc);
    assert!(mc.state_storage.seen.contains_key(&fp));
}

#[test]
fn full_state_admit_new_state_returns_entry() {
    let module = minimal_module();
    let config = minimal_config();
    let mut mc = ModelChecker::new(&module, &config);
    mc.set_store_states(true);

    let fp = Fingerprint(42);
    let state = ArrayState::from_values(vec![Value::int(7)]);

    let mut storage = FullStateStorage;
    let entry = storage
        .admit_successor(fp, state, None, None, 0, &mut mc)
        .unwrap();
    assert!(entry.is_some());
    let (ret_fp, ret_depth) = entry.unwrap();
    assert_eq!(ret_fp, fp);
    assert_eq!(ret_depth, 0);
}

#[test]
fn full_state_admit_duplicate_returns_none() {
    let module = minimal_module();
    let config = minimal_config();
    let mut mc = ModelChecker::new(&module, &config);
    mc.set_store_states(true);

    let fp = Fingerprint(42);

    let mut storage = FullStateStorage;
    let first = storage
        .admit_successor(
            fp,
            ArrayState::from_values(vec![Value::int(7)]),
            None,
            None,
            0,
            &mut mc,
        )
        .unwrap();
    assert!(first.is_some());

    let second = storage
        .admit_successor(
            fp,
            ArrayState::from_values(vec![Value::int(7)]),
            None,
            None,
            1,
            &mut mc,
        )
        .unwrap();
    assert!(second.is_none());
}

#[test]
fn full_state_admit_duplicate_uses_current_parent_payload_when_dequeued() {
    let module = minimal_module();
    let config = minimal_config();
    let mut mc = ModelChecker::new(&module, &config);
    mc.set_store_states(true);

    let fp = Fingerprint(42);
    let mut storage = FullStateStorage;
    let first = storage
        .admit_successor(
            fp,
            ArrayState::from_values(vec![Value::int(7)]),
            None,
            None,
            0,
            &mut mc,
        )
        .unwrap();
    assert!(first.is_some());

    let (_current_fp, current, _depth) = storage.dequeue((fp, 0), &mut mc).unwrap().unwrap();
    assert!(!mc.state_storage.seen.contains_key(&fp));

    let duplicate = storage
        .admit_successor(
            fp,
            current.clone(),
            Some(fp),
            Some((fp, &current)),
            1,
            &mut mc,
        )
        .unwrap();
    assert!(duplicate.is_none());
    assert_eq!(mc.states_count(), 1);

    storage.return_current(fp, current, &mut mc);
}

#[test]
fn full_state_use_diffs_true_by_default() {
    let module = minimal_module();
    let config = minimal_config();
    let mc = ModelChecker::new(&module, &config);

    let storage = FullStateStorage;
    // Default: no VIEW, no symmetry, no liveness → true
    // (unless TY_FORCE_NO_DIFFS env var is set)
    if !crate::check::debug::force_no_diffs() {
        assert!(storage.use_diffs(&mc));
    }
}

#[test]
fn full_state_cache_full_liveness_noop_when_not_caching() {
    let module = minimal_module();
    let config = minimal_config();
    let mut mc = ModelChecker::new(&module, &config);
    // Default: cache_for_liveness is false

    let storage = FullStateStorage;
    let successors = vec![(
        ArrayState::from_values(vec![Value::int(1)]),
        Fingerprint(10),
    )];
    let result = storage.cache_full_liveness(Fingerprint(1), &successors, &mut mc);
    assert!(result.is_ok());
}

// -----------------------------------------------------------------------
// FingerprintOnlyStorage tests
// -----------------------------------------------------------------------

#[test]
fn fp_only_dequeue_owned_entry_returns_state() {
    let module = minimal_module();
    let config = minimal_config();
    let mut mc = ModelChecker::new(&module, &config);

    let fp = Fingerprint(42);
    let state = ArrayState::from_values(vec![Value::int(7)]);
    let entry = NoTraceQueueEntry::Owned {
        state: state.clone(),
        fp,
    };

    let bulk = BulkStateStorage::empty(1);
    let mut storage = FingerprintOnlyStorage::new(bulk, 1);
    let result = storage.dequeue((entry, 3, 0), &mut mc).unwrap().unwrap();
    assert_eq!(result.0, fp);
    assert_eq!(result.2, 3);
}

#[test]
fn fp_only_dequeue_sets_parent_trace_loc() {
    let module = minimal_module();
    let config = minimal_config();
    let mut mc = ModelChecker::new(&module, &config);

    let fp = Fingerprint(42);
    let state = ArrayState::from_values(vec![Value::int(7)]);
    let entry = NoTraceQueueEntry::Owned { state, fp };
    let trace_loc = 12345u64;

    let bulk = BulkStateStorage::empty(1);
    let mut storage = FingerprintOnlyStorage::new(bulk, 1);
    storage.dequeue((entry, 0, trace_loc), &mut mc).unwrap();
    assert_eq!(mc.trace.current_parent_trace_loc, Some(trace_loc));
}

#[test]
fn fp_only_return_current_is_noop() {
    let module = minimal_module();
    let config = minimal_config();
    let mut mc = ModelChecker::new(&module, &config);

    let bulk = BulkStateStorage::empty(1);
    let mut storage = FingerprintOnlyStorage::new(bulk, 1);

    let states_before = mc.stats.states_found;
    let phantoms_before = mc.stats.phantom_dequeues;

    // Should not panic or modify any state
    storage.return_current(
        Fingerprint(42),
        ArrayState::from_values(vec![Value::int(7)]),
        &mut mc,
    );

    assert_eq!(
        mc.stats.states_found, states_before,
        "return_current should not change states_found"
    );
    assert_eq!(
        mc.stats.phantom_dequeues, phantoms_before,
        "return_current should not change phantom_dequeues"
    );
}

#[test]
fn fp_only_admit_new_state_returns_entry() {
    let module = minimal_module();
    let config = minimal_config();
    let mut mc = ModelChecker::new(&module, &config);

    let fp = Fingerprint(42);
    let state = ArrayState::from_values(vec![Value::int(7)]);

    let bulk = BulkStateStorage::empty(1);
    let mut storage = FingerprintOnlyStorage::new(bulk, 1);
    let entry = storage
        .admit_successor(fp, state, None, None, 0, &mut mc)
        .unwrap();
    assert!(entry.is_some());
    let (queue_entry, depth, _trace_loc) = entry.unwrap();
    assert_eq!(depth, 0);
    assert!(matches!(queue_entry, NoTraceQueueEntry::Owned { fp: f, .. } if f == fp));
    assert!(
        !mc.state_storage.seen.contains_key(&fp),
        "fp-only admission should retain compact witnesses, not ArrayState payloads"
    );
}

#[test]
fn fp_only_admit_payload_confirmed_duplicate_returns_none() {
    let module = minimal_module();
    let config = minimal_config();
    let mut mc = ModelChecker::new(&module, &config);

    let fp = Fingerprint(42);

    let bulk = BulkStateStorage::empty(1);
    let mut storage = FingerprintOnlyStorage::new(bulk, 1);
    let first = storage
        .admit_successor(
            fp,
            ArrayState::from_values(vec![Value::int(7)]),
            None,
            None,
            0,
            &mut mc,
        )
        .unwrap();
    assert!(first.is_some());

    let duplicate = storage
        .admit_successor(
            fp,
            ArrayState::from_values(vec![Value::int(7)]),
            None,
            None,
            1,
            &mut mc,
        )
        .unwrap();
    assert!(duplicate.is_none());
    assert!(
        !mc.state_storage.seen.contains_key(&fp),
        "fp-only duplicate witnesses should not populate full-state storage"
    );
}

#[test]
fn fp_only_batch_admit_storage_fault_stops_before_faulted_candidate() {
    let module = minimal_module();
    let config = minimal_config();
    let mut mc = ModelChecker::new(&module, &config);
    mc.state_storage.seen_fps = Arc::new(FaultAfterFirstInsertSet::new());

    let bulk = BulkStateStorage::empty(1);
    let mut storage = FingerprintOnlyStorage::new(bulk, 1);
    let mut candidates = vec![
        BfsAdmissionCandidate {
            fp: Fingerprint(42),
            state: ArrayState::from_values(vec![Value::int(7)]),
            parent_fp: Some(Fingerprint(1)),
            depth: 1,
        },
        BfsAdmissionCandidate {
            fp: Fingerprint(43),
            state: ArrayState::from_values(vec![Value::int(8)]),
            parent_fp: Some(Fingerprint(1)),
            depth: 1,
        },
        BfsAdmissionCandidate {
            fp: Fingerprint(44),
            state: ArrayState::from_values(vec![Value::int(9)]),
            parent_fp: Some(Fingerprint(1)),
            depth: 1,
        },
    ];
    let mut result = BfsBatchAdmissionResult::with_capacity(candidates.len());

    storage.admit_successor_batch_into(&mut candidates, None, &mut result, &mut mc);

    assert_eq!(result.entries.len(), 1);
    assert!(result.entries[0].entry.is_some());
    assert!(result.fault.is_some());
    assert!(candidates.is_empty());
    assert_eq!(mc.states_count(), 1);
}

#[test]
fn fp_only_admit_same_fingerprint_different_payload_fails_closed() {
    let module = minimal_module();
    let config = minimal_config();
    let mut mc = ModelChecker::new(&module, &config);

    let fp = Fingerprint(42);

    let bulk = BulkStateStorage::empty(1);
    let mut storage = FingerprintOnlyStorage::new(bulk, 1);
    let first = storage
        .admit_successor(
            fp,
            ArrayState::from_values(vec![Value::int(7)]),
            None,
            None,
            0,
            &mut mc,
        )
        .unwrap();
    assert!(first.is_some());

    let err = match storage.admit_successor(
        fp,
        ArrayState::from_values(vec![Value::int(8)]),
        None,
        None,
        1,
        &mut mc,
    ) {
        Err(err) => err,
        Ok(_) => panic!("fp-only duplicate with mismatched payload must fail closed"),
    };
    match err {
        CheckResult::Error { error, .. } => {
            let rendered = error.to_string();
            // Part of #4451 follow-up: the prepared admission backend renders
            // the fail-closed collision with `reason_code=canonical_payload_mismatch`
            // (and `payload_witness=tla_array_fp64` for ArrayState-domain witnesses).
            assert!(
                rendered.contains("reason_code=canonical_payload_mismatch"),
                "unexpected error: {rendered}"
            );
        }
        other => panic!("expected CheckResult::Error, got {other:?}"),
    }
}

#[test]
fn fp_only_admit_current_payload_confirmed_duplicate_returns_none() {
    let module = minimal_module();
    let config = minimal_config();
    let mut mc = ModelChecker::new(&module, &config);

    let fp = Fingerprint(42);
    let state = ArrayState::from_values(vec![Value::int(7)]);

    let bulk = BulkStateStorage::empty(1);
    let mut storage = FingerprintOnlyStorage::new(bulk, 1);
    let first = storage
        .admit_successor(fp, state.clone(), None, None, 0, &mut mc)
        .unwrap();
    assert!(first.is_some());

    let second = storage
        .admit_successor(fp, state.clone(), Some(fp), Some((fp, &state)), 1, &mut mc)
        .unwrap();
    assert!(second.is_none());
}

#[test]
fn fp_only_admit_current_payload_confirmed_duplicate_without_stored_witness_returns_none() {
    let module = minimal_module();
    let config = minimal_config();
    let mut mc = ModelChecker::new(&module, &config);

    let fp = Fingerprint(43);
    let state = ArrayState::from_values(vec![Value::int(7)]);
    mc.mark_state_seen_fp_only_checked(fp, None, 0).unwrap();

    let bulk = BulkStateStorage::empty(1);
    let mut storage = FingerprintOnlyStorage::new(bulk, 1);
    let duplicate = storage
        .admit_successor(fp, state.clone(), Some(fp), Some((fp, &state)), 1, &mut mc)
        .unwrap();
    assert!(duplicate.is_none());
    assert!(!mc.state_storage.seen.contains_key(&fp));
}

#[test]
fn fp_only_admit_duplicate_can_use_resident_initial_payload_witness() {
    let module = minimal_module();
    let config = minimal_config();
    let mut mc = ModelChecker::new(&module, &config);

    let fp = Fingerprint(44);
    let initial = ArrayState::from_values(vec![Value::int(7)]);
    assert!(mc
        .mark_state_seen_checked_with_current(fp, &initial, None, 0, None)
        .unwrap());

    let bulk = BulkStateStorage::empty(1);
    let mut storage = FingerprintOnlyStorage::new(bulk, 1);
    let duplicate = storage
        .admit_successor(
            fp,
            ArrayState::from_values(vec![Value::int(7)]),
            None,
            None,
            1,
            &mut mc,
        )
        .unwrap();
    assert!(duplicate.is_none());
    assert_eq!(mc.states_count(), 1);
}

#[test]
fn fp_only_view_duplicate_uses_view_witness_without_full_state_resident() {
    let module = view_module();
    let config = view_config();
    let mut mc = ModelChecker::new(&module, &config);
    mc.compiled.cached_view_name = Some("View".to_string());

    let fp = Fingerprint(45);
    let bulk = BulkStateStorage::empty(2);
    let mut storage = FingerprintOnlyStorage::new(bulk, 2);

    let first = storage
        .admit_successor(
            fp,
            ArrayState::from_values(vec![Value::int(7), Value::int(0)]),
            None,
            None,
            0,
            &mut mc,
        )
        .unwrap();
    assert!(first.is_some());
    assert!(
        !mc.state_storage.seen.contains_key(&fp),
        "VIEW fp-only admission should keep a VIEW payload witness, not a resident ArrayState"
    );

    let duplicate_same_view = storage
        .admit_successor(
            fp,
            ArrayState::from_values(vec![Value::int(7), Value::int(1)]),
            None,
            None,
            1,
            &mut mc,
        )
        .unwrap();
    assert!(duplicate_same_view.is_none());
    assert!(!mc.state_storage.seen.contains_key(&fp));

    let mismatch = storage.admit_successor(
        fp,
        ArrayState::from_values(vec![Value::int(8), Value::int(1)]),
        None,
        None,
        1,
        &mut mc,
    );
    assert!(
        mismatch.is_err(),
        "same fingerprint with a different VIEW payload must fail closed"
    );
}

#[test]
fn fp_only_view_duplicate_borrows_resident_once_then_uses_view_witness() {
    let module = view_module();
    let config = view_config();
    let mut mc = ModelChecker::new(&module, &config);
    mc.compiled.cached_view_name = Some("View".to_string());

    let fp = Fingerprint(46);
    let resident = ArrayState::from_values(vec![Value::int(7), Value::int(0)]);
    assert!(mc
        .mark_state_seen_checked_with_current(fp, &resident, None, 0, None)
        .unwrap());
    assert!(mc.state_storage.seen.contains_key(&fp));

    let bulk = BulkStateStorage::empty(2);
    let mut storage = FingerprintOnlyStorage::new(bulk, 2);
    let duplicate_from_resident = storage
        .admit_successor(
            fp,
            ArrayState::from_values(vec![Value::int(7), Value::int(1)]),
            None,
            None,
            1,
            &mut mc,
        )
        .unwrap();
    assert!(duplicate_from_resident.is_none());

    mc.state_storage.seen.clear();
    let duplicate_from_witness = storage
        .admit_successor(
            fp,
            ArrayState::from_values(vec![Value::int(7), Value::int(2)]),
            None,
            None,
            2,
            &mut mc,
        )
        .unwrap();
    assert!(duplicate_from_witness.is_none());
}

#[test]
fn fp_only_use_diffs_true_by_default() {
    let module = minimal_module();
    let config = minimal_config();
    let mc = ModelChecker::new(&module, &config);

    let bulk = BulkStateStorage::empty(1);
    let storage = FingerprintOnlyStorage::new(bulk, 1);
    // Default: no VIEW, no symmetry, cache_for_liveness=false → true
    if !crate::check::debug::force_no_diffs() {
        assert!(storage.use_diffs(&mc));
    }
}

#[test]
fn fp_only_cache_diff_liveness_is_noop() {
    let module = minimal_module();
    let config = minimal_config();
    let mut mc = ModelChecker::new(&module, &config);

    let bulk = BulkStateStorage::empty(1);
    let storage = FingerprintOnlyStorage::new(bulk, 1);
    // Should succeed without error (no-op, doesn't store anything)
    let result = storage.cache_diff_liveness(Fingerprint(1), Some(vec![Fingerprint(2)]), &mut mc);
    assert!(result.is_ok());
}

#[test]
fn fp_only_cache_full_liveness_noop_when_not_caching() {
    let module = minimal_module();
    let config = minimal_config();
    let mut mc = ModelChecker::new(&module, &config);

    let bulk = BulkStateStorage::empty(1);
    let storage = FingerprintOnlyStorage::new(bulk, 1);
    let successors = vec![(
        ArrayState::from_values(vec![Value::int(1)]),
        Fingerprint(10),
    )];
    let result = storage.cache_full_liveness(Fingerprint(1), &successors, &mut mc);
    assert!(result.is_ok());
}
