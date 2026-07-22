// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Unit tests for the fingerprint-keyed implied-action transition cache
//! (zero-arg derived-operator memoization across transitions).
//!
//! Soundness contract under test:
//! * store eligibility is fail-closed (single-sided state deps only, no
//!   locals, no TLCGet("level"), no taints);
//! * every hit re-validates all recorded dep values against the bound arrays
//!   (a fingerprint collision degrades to a miss, never a wrong value);
//! * mode normalization: entries stored from `Next`-mode evaluations validate
//!   and propagate correctly when reused, and vice versa.

use super::*;
use crate::cache::dep_tracking::VarDepMap;
use crate::cache::zero_arg_cache::{
    zero_arg_transition_cache_entry_count, zero_arg_transition_insert, zero_arg_transition_lookup,
    TransitionPartition, ZeroArgTransitionCacheKey,
};
use crate::cache::{CachedOpResult, OpEvalDeps};
use crate::eval_ident_zero_arg::{
    lazy_domain_is_small_finite, normalize_transition_deps, transition_deps_eligible,
    transition_deps_for_mode, transition_entry_valid, transition_value_captures_state,
};

fn state_deps(entries: &[(tla_core::VarIndex, Value)]) -> OpEvalDeps {
    OpEvalDeps {
        state: VarDepMap::from_entries(entries),
        ..Default::default()
    }
}

fn next_deps(entries: &[(tla_core::VarIndex, Value)]) -> OpEvalDeps {
    OpEvalDeps {
        next: VarDepMap::from_entries(entries),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Store eligibility (fail-closed)
// ---------------------------------------------------------------------------

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_transition_eligibility_accepts_single_sided_state_reads() {
    let mut ctx = EvalCtx::new();
    ctx.register_var("x");
    let x = ctx.var_registry().get("x").expect("registered");

    // Current mode: state-only reads are eligible.
    assert!(transition_deps_eligible(
        &state_deps(&[(x, Value::int(1))]),
        false
    ));
    // Next mode: next-only reads are eligible.
    assert!(transition_deps_eligible(
        &next_deps(&[(x, Value::int(1))]),
        true
    ));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_transition_eligibility_fail_closed() {
    let mut ctx = EvalCtx::new();
    ctx.register_var("x");
    let x = ctx.var_registry().get("x").expect("registered");

    // Empty deps (constant op): not a transition-cache concern.
    assert!(!transition_deps_eligible(&OpEvalDeps::default(), false));
    assert!(!transition_deps_eligible(&OpEvalDeps::default(), true));

    // Wrong side for the mode.
    assert!(!transition_deps_eligible(
        &state_deps(&[(x, Value::int(1))]),
        true
    ));
    assert!(!transition_deps_eligible(
        &next_deps(&[(x, Value::int(1))]),
        false
    ));

    // Mixed sides (action-level operator) — never eligible.
    let mixed = OpEvalDeps {
        state: VarDepMap::from_entries(&[(x, Value::int(1))]),
        next: VarDepMap::from_entries(&[(x, Value::int(2))]),
        ..Default::default()
    };
    assert!(!transition_deps_eligible(&mixed, false));
    assert!(!transition_deps_eligible(&mixed, true));

    // Local reads.
    let mut with_local = state_deps(&[(x, Value::int(1))]);
    with_local.record_local(tla_core::name_intern::intern_name("k"), &Value::int(0));
    assert!(!transition_deps_eligible(&with_local, false));

    // TLCGet("level") dependence — value varies per BFS level, not per state.
    let mut with_level = state_deps(&[(x, Value::int(1))]);
    with_level.record_tlc_level(3);
    assert!(!transition_deps_eligible(&with_level, false));

    // Inconsistency and INSTANCE lazy-read taints.
    let mut inconsistent = state_deps(&[(x, Value::int(1))]);
    inconsistent.inconsistent = true;
    assert!(!transition_deps_eligible(&inconsistent, false));

    let mut tainted = state_deps(&[(x, Value::int(1))]);
    tainted.instance_lazy_read = true;
    assert!(!transition_deps_eligible(&tainted, false));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_transition_value_captures_state_rejects_lazy_funcs() {
    // Plain data values are storable.
    assert!(!transition_value_captures_state(&Value::int(42)));
    assert!(!transition_value_captures_state(&Value::Bool(true)));
    // LazyFunc capture detection is delegated to captured_state(); a
    // full LazyFunc construction test lives in the end-to-end checker test
    // (materialization converts small lazy funcs to eager Funcs before the
    // store, so no capturing value can reach the partition).
}

// ---------------------------------------------------------------------------
// Validation by dep values (fingerprint collisions degrade to misses)
// ---------------------------------------------------------------------------

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_transition_entry_validates_against_bound_arrays_per_mode() {
    let mut ctx = EvalCtx::new();
    ctx.register_var("x");
    let x = ctx.var_registry().get("x").expect("registered");

    // Entry as stored from a Current-mode evaluation of a derived op that
    // read x = 7 (deps normalized to the state side).
    let entry = CachedOpResult {
        value: Value::int(700),
        deps: normalize_transition_deps(state_deps(&[(x, Value::int(7))]), false),
    };

    // Parent bound with x = 7 → valid in Current mode.
    let parent = vec![Value::int(7)];
    let succ = vec![Value::int(9)];
    ctx.bind_state_array(&parent);
    ctx.bind_next_state_array(&succ);
    assert!(transition_entry_valid(&ctx, &entry, false));
    // Successor has x = 9 → the same entry must NOT validate in Next mode
    // (this is exactly the fingerprint-collision fail-closed path).
    assert!(!transition_entry_valid(&ctx, &entry, true));

    // Entry stored from a Next-mode evaluation that read x' = 9.
    let entry_next = CachedOpResult {
        value: Value::int(900),
        deps: normalize_transition_deps(next_deps(&[(x, Value::int(9))]), true),
    };
    assert!(transition_entry_valid(&ctx, &entry_next, true));
    assert!(!transition_entry_valid(&ctx, &entry_next, false));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_transition_entry_invalid_without_bound_array() {
    let mut ctx = EvalCtx::new();
    ctx.register_var("x");
    let x = ctx.var_registry().get("x").expect("registered");

    let entry = CachedOpResult {
        value: Value::int(700),
        deps: normalize_transition_deps(state_deps(&[(x, Value::int(7))]), false),
    };

    // No arrays bound: fail-closed in both modes.
    assert!(!transition_entry_valid(&ctx, &entry, false));
    assert!(!transition_entry_valid(&ctx, &entry, true));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_transition_deps_propagate_on_the_mode_side() {
    let mut ctx = EvalCtx::new();
    ctx.register_var("x");
    let x = ctx.var_registry().get("x").expect("registered");

    let stored = normalize_transition_deps(next_deps(&[(x, Value::int(9))]), true);
    // Stored normalized on the state side.
    assert!(!stored.state.is_empty());
    assert!(stored.next.is_empty());

    // Hit in Current mode → propagate as state reads.
    let cur = transition_deps_for_mode(&stored, false);
    assert!(!cur.state.is_empty());
    assert!(cur.next.is_empty());

    // Hit in Next mode → propagate as next reads.
    let nxt = transition_deps_for_mode(&stored, true);
    assert!(nxt.state.is_empty());
    assert!(!nxt.next.is_empty());
}

// ---------------------------------------------------------------------------
// Partition mechanics
// ---------------------------------------------------------------------------

fn transition_partition_key(fp: u64) -> ZeroArgTransitionCacheKey {
    (
        1,
        0,
        0,
        tla_core::name_intern::intern_name("transition_partition_test_op"),
        0,
        fp,
    )
}

fn transition_partition_entry(value: i64) -> CachedOpResult {
    CachedOpResult {
        value: Value::int(value),
        deps: OpEvalDeps::default(),
    }
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_transition_partition_rotation_recycles_maps_and_preserves_generations() {
    let mut partition = TransitionPartition::with_generation_cap_for_test(2, true);
    let keys: Vec<_> = (1..=5).map(transition_partition_key).collect();

    partition.insert(keys[0], transition_partition_entry(1));
    partition.insert(keys[1], transition_partition_entry(2));
    // An overwrite in current must not rotate or grow the generation.
    partition.insert(keys[0], transition_partition_entry(11));
    assert_eq!(partition.generation_lens_for_test(), (2, 0));
    // First bounded rotation: {1,2} becomes previous and 3 is current.
    partition.insert(keys[2], transition_partition_entry(3));
    assert_eq!(partition.generation_lens_for_test(), (1, 2));
    assert_eq!(
        partition.get(&keys[0]).map(|e| &e.value),
        Some(&Value::int(11))
    );
    assert_eq!(
        partition.get(&keys[2]).map(|e| &e.value),
        Some(&Value::int(3))
    );

    // Exercise the mutable lookup in both generations.
    partition.get_mut(&keys[0]).expect("previous hit").value = Value::int(10);
    partition.get_mut(&keys[2]).expect("current hit").value = Value::int(30);

    partition.insert(keys[3], transition_partition_entry(4));
    let capacities_before_second_rotation = partition.generation_capacities_for_test();
    assert!(capacities_before_second_rotation.0 >= 2);
    assert!(capacities_before_second_rotation.1 >= 2);

    // Second rotation evicts old previous {1,2}, retains old current {3,4},
    // and reuses the old previous map allocation for current {5}.
    partition.insert(keys[4], transition_partition_entry(5));
    assert_eq!(partition.generation_lens_for_test(), (1, 2));
    assert!(partition.get(&keys[0]).is_none());
    assert_eq!(
        partition.get(&keys[2]).map(|e| &e.value),
        Some(&Value::int(30))
    );
    assert_eq!(
        partition.get(&keys[4]).map(|e| &e.value),
        Some(&Value::int(5))
    );
    assert_eq!(
        partition.generation_capacities_for_test(),
        (
            capacities_before_second_rotation.1,
            capacities_before_second_rotation.0,
        )
    );
    assert!(partition.len() <= 4);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_transition_partition_recycle_kill_switch_keeps_legacy_eviction() {
    let mut partition = TransitionPartition::with_generation_cap_for_test(2, false);
    let keys: Vec<_> = (1..=5).map(transition_partition_key).collect();

    for (idx, key) in keys.iter().enumerate() {
        partition.insert(*key, transition_partition_entry(idx as i64 + 1));
    }

    assert_eq!(partition.generation_lens_for_test(), (1, 2));
    assert!(partition.get(&keys[0]).is_none());
    assert_eq!(
        partition.get(&keys[2]).map(|e| &e.value),
        Some(&Value::int(3))
    );
    assert_eq!(
        partition.get(&keys[4]).map(|e| &e.value),
        Some(&Value::int(5))
    );
    assert!(partition.len() <= 4);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_transition_partition_insert_lookup_and_reset() {
    clear_for_test_reset();
    let mut ctx = EvalCtx::new();
    ctx.register_var("x");
    let x = ctx.var_registry().get("x").expect("registered");

    let key = (
        1u64,
        0u64,
        0u64,
        tla_core::name_intern::intern_name("op"),
        0u32,
        0xDEAD_BEEFu64,
    );
    let entry = CachedOpResult {
        value: Value::int(700),
        deps: normalize_transition_deps(state_deps(&[(x, Value::int(7))]), false),
    };
    zero_arg_transition_insert(key, entry);
    assert_eq!(zero_arg_transition_cache_entry_count(), 1);

    // Validator accepting → hit.
    let hit = zero_arg_transition_lookup(&key, |_| true);
    assert_eq!(hit.map(|e| e.value), Some(Value::int(700)));

    // Validator rejecting (dep mismatch) → miss.
    assert!(zero_arg_transition_lookup(&key, |_| false).is_none());

    // Different fingerprint → miss.
    let other_key = (key.0, key.1, key.2, key.3, key.4, 0xF00Du64);
    assert!(zero_arg_transition_lookup(&other_key, |_| true).is_none());

    // Test/run reset clears the partition.
    clear_for_test_reset();
    assert_eq!(zero_arg_transition_cache_entry_count(), 0);
}

// ---------------------------------------------------------------------------
// Materialization guard
// ---------------------------------------------------------------------------

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_lazy_domain_small_finite_guard() {
    use crate::value::LazyDomain;

    // Small finite general domain qualifies.
    let small = LazyDomain::General(Box::new(Value::set(vec![
        Value::int(0),
        Value::int(1),
        Value::int(2),
    ])));
    assert!(lazy_domain_is_small_finite(&small));

    // Infinite built-in domains never qualify.
    assert!(!lazy_domain_is_small_finite(&LazyDomain::Nat));
    assert!(!lazy_domain_is_small_finite(&LazyDomain::Int));
    assert!(!lazy_domain_is_small_finite(&LazyDomain::String));

    // Large finite domain (> 64) does not qualify.
    let big = LazyDomain::General(Box::new(Value::set(
        (0..100).map(Value::int).collect::<Vec<_>>(),
    )));
    assert!(!lazy_domain_is_small_finite(&big));
}
