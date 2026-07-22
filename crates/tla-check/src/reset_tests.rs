// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for `reset_global_state()`.
//!
//! Extracted from `lib.rs` test module. Part of #3608.

use super::reset_global_state_unchecked;
use crate::intern::{get_interner, wait_for_no_active_runs};
use crate::{guard_error_stats, intern, value, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static RESET_GLOBAL_STATE_TEST_ID: AtomicU64 = AtomicU64::new(1);

struct ResetSnapshots {
    string_key: String,
    model_value_name: String,
    name_key: String,
    string_arc: tla_value::Rp<str>,
    set_arc: tla_value::Rp<[Value]>,
    int_func: value::IntIntervalFunc,
}

fn next_reset_test_id() -> u64 {
    RESET_GLOBAL_STATE_TEST_ID.fetch_add(1, Ordering::Relaxed)
}

fn populate_reset_targets(test_id: u64) -> (ResetSnapshots, intern::ModelCheckRunGuard) {
    let unique_small_int = 1_000_000_000_i64
        + i64::try_from(test_id).expect("reset_global_state test id should fit in i64");
    let string_key = format!("test_reset_string_{test_id}");
    let model_value_name = format!("test_reset_model_value_{test_id}");
    let name_key = format!("test_reset_name_{test_id}");

    // 1) Value interner (tla-check/intern)
    let interner = get_interner();
    let value_entries_before = interner.len();
    interner.intern(Value::int(unique_small_int));
    assert!(
        interner.len() > value_entries_before,
        "value interner entry count should increase after inserting unique value"
    );

    // 2) String interner (tla-value/strings)
    let string_before = value::intern_string(&string_key);
    let string_before_again = value::intern_string(&string_key);
    assert!(
        tla_value::Rp::ptr_eq(&string_before, &string_before_again),
        "string interner should deduplicate before reset"
    );

    // 3) Set interner (tla-value/intern_tables)
    let set_before_1 = Value::set([Value::int(1), Value::int(2)])
        .as_set()
        .expect("Value::set should return Value::Set")
        .to_arc_slice();
    let set_before_2 = Value::set([Value::int(1), Value::int(2)])
        .as_set()
        .expect("Value::set should return Value::Set")
        .to_arc_slice();
    assert!(
        tla_value::Rp::ptr_eq(&set_before_1, &set_before_2),
        "set interner should deduplicate before reset"
    );

    // 4) Int-func interner (tla-value/intern_tables)
    // IntIntervalFunc::except() interns small arrays via intern_int_func_array().
    let int_func_before_1 = value::IntIntervalFunc::new(1, 2, vec![Value::int(0), Value::int(0)])
        .except(Value::int(1), Value::int(7));
    let int_func_before_2 = value::IntIntervalFunc::new(1, 2, vec![Value::int(0), Value::int(0)])
        .except(Value::int(1), Value::int(7));
    assert!(
        int_func_before_1.values().as_ptr() == int_func_before_2.values().as_ptr(),
        "int-func interner should deduplicate before reset"
    );

    // 5) Model value registry (tla-value/model_values)
    let model_values_before = value::model_value_count();
    let _ = Value::try_model_value(&model_value_name).unwrap();
    assert!(
        value::model_value_count() > model_values_before,
        "model value registry count should increase after inserting unique model value"
    );

    // 6) Name interner (tla-core/name_intern)
    let names_before = tla_core::interned_name_count();
    let _ = tla_core::intern_name(&name_key);
    assert!(
        tla_core::interned_name_count() > names_before,
        "name interner count should increase after inserting unique name"
    );

    // Verify steps 1-6 populated the value interner before the step-7
    // lifecycle transition. begin_model_check_run() clears run-scoped
    // interners on the 0-to-1 refcount transition.
    assert!(
        !interner.is_empty(),
        "value interner should have entries before begin_model_check_run"
    );

    // 7) Part of #3384, Part of #3334: seed run-scoped state that
    // reset_global_state() must clear between independent checks.
    let model_check_run = intern::ModelCheckRunGuard::begin();
    let _run_guard =
        tla_value::ParallelValueInternRunGuard::new(tla_value::SharedValueCacheMode::Readonly);
    {
        let _worker_scope = tla_value::WorkerInternGuard::new();
    }
    guard_error_stats::record_suppressed_action_level_error();

    // Re-populate the value interner after begin_model_check_run cleared
    // it, so reset_global_state() has a non-empty interner to clear.
    interner.intern(Value::int(unique_small_int + 1));
    assert!(
        !interner.is_empty(),
        "value interner should have entries after re-populate"
    );

    (
        ResetSnapshots {
            string_key,
            model_value_name,
            name_key,
            string_arc: string_before,
            set_arc: set_before_1,
            int_func: int_func_before_1,
        },
        model_check_run,
    )
}

fn assert_all_tables_cleared_after_reset(snapshots: &ResetSnapshots) {
    // 1) Value interner cleared
    assert_eq!(
        get_interner().len(),
        0,
        "value interner should be empty after reset"
    );

    // 2) Model value registry cleared
    assert_eq!(
        value::model_value_count(),
        0,
        "model value count should be 0 after reset"
    );

    // 3) Name interner cleared
    assert_eq!(
        tla_core::interned_name_count(),
        0,
        "name interner should be empty after reset"
    );

    // 4) String interner cleared (new interned value should not reuse old Arc)
    let string_after = value::intern_string(&snapshots.string_key);
    assert!(
        !tla_value::Rp::ptr_eq(&snapshots.string_arc, &string_after),
        "string interner should not reuse pre-reset Arc after reset"
    );

    // 5) Set interner cleared (new equivalent set should not reuse old Arc)
    let set_after = Value::set([Value::int(1), Value::int(2)])
        .as_set()
        .expect("Value::set should return Value::Set")
        .to_arc_slice();
    assert!(
        !tla_value::Rp::ptr_eq(&snapshots.set_arc, &set_after),
        "set interner should not reuse pre-reset Arc after reset"
    );

    // 6) Int-func interner cleared (new equivalent int-func should not reuse old Arc)
    let int_func_after = value::IntIntervalFunc::new(1, 2, vec![Value::int(0), Value::int(0)])
        .except(Value::int(1), Value::int(7));
    assert!(
        snapshots.int_func.values().as_ptr() != int_func_after.values().as_ptr(),
        "int-func interner should not reuse pre-reset Arc after reset"
    );

    // Model value/name interning should restart from an empty registry after reset.
    let _ = Value::try_model_value(&snapshots.model_value_name).unwrap();
    assert_eq!(
        value::model_value_count(),
        1,
        "model value registry should restart from empty after reset"
    );
    let _ = tla_core::intern_name(&snapshots.name_key);
    assert_eq!(
        tla_core::interned_name_count(),
        1,
        "name interner should restart from empty after reset"
    );

    // 7) Part of #3384: reset must clear the parallel value-intern cleanup
    // boundary even if a prior run leaked it.
    assert!(
        !tla_value::parallel_readonly_value_caches_active(),
        "parallel read-only cache mode should be disabled after reset"
    );
    let worker_scope_after_reset = std::panic::catch_unwind(tla_value::WorkerInternGuard::new);
    assert!(
        worker_scope_after_reset.is_err(),
        "worker intern scope should fail after reset because the frozen snapshot must be cleared"
    );

    // A leaked ACTIVE_MODEL_CHECK_RUNS counter must not suppress the next
    // run-start clear of run-scoped interners.
    let run_scoped_value = Value::int(9_999_999_999_i64);
    get_interner().intern(run_scoped_value);
    let run_scoped_string_key = format!("{}_run_scoped", snapshots.string_key);
    let run_scoped_string_before = value::intern_string(&run_scoped_string_key);
    intern::begin_model_check_run();
    assert_eq!(
        get_interner().len(),
        0,
        "begin_model_check_run after reset should clear leaked value-interner state"
    );
    let run_scoped_string_after = value::intern_string(&run_scoped_string_key);
    assert!(
        !tla_value::Rp::ptr_eq(&run_scoped_string_before, &run_scoped_string_after),
        "begin_model_check_run after reset should clear leaked string-interner state"
    );
    intern::end_model_check_run();

    // 8) Part of #3025: OP_RESULT_CACHE removed (zero insertions).
    // The old op_result_cache_len() assertion was vacuously true. Run reset
    // correctness is verified by the clear_for_test_reset() path through
    // clear_run_reset_impl(), which clears all active caches.
}

/// Evaluate an EXCEPT expression whose replacement depends on the special `@`
/// binding.  This crosses the public tla-eval entry point, so it catches stale
/// pre-interned binding IDs rather than merely inspecting interner bookkeeping.
fn eval_except_at_binding() -> Value {
    let module = crate::test_support::parse_module(
        "---- MODULE ResetExceptAt ----\n\
         EXTENDS Integers\n\
         Op == [[x |-> 1] EXCEPT !.x = @ + 1].x\n\
         ====\n",
    );
    let body = module
        .units
        .iter()
        .find_map(|unit| match &unit.node {
            tla_core::ast::Unit::Operator(def) if def.name.node == "Op" => Some(&def.body),
            _ => None,
        })
        .expect("Op must lower");
    crate::eval(&crate::EvalCtx::new(), body).expect("EXCEPT @ expression must evaluate")
}

/// Verify that `reset_global_state()` clears all global interning tables.
/// Part of #1331: memory safety audit.
/// Part of #4067: Wait for concurrent model check runs (e.g., real_disruptor,
/// ~100s each) before acquiring the interner lock. The unchecked reset clears
/// everything including the name interner, which would cause index-out-of-bounds
/// panics in concurrent tests if called while they're active. The 600s timeout
/// accommodates: wait_for_no_active_runs (up to 120s) + test body + cleanup.
#[test]
fn test_reset_global_state_clears_all_tables() {
    let _quiescence = crate::test_utils::acquire_model_check_quiescence_lock();
    test_reset_global_state_clears_all_tables_body();
}

#[cfg_attr(test, ntest::timeout(600_000))]
fn test_reset_global_state_clears_all_tables_body() {
    // Part of #4067: Wait for ALL concurrent model check runs to finish
    // BEFORE acquiring the interner lock and calling unchecked reset.
    // This ensures the name interner clear doesn't corrupt concurrent tests.
    wait_for_no_active_runs();
    let _serial = crate::test_utils::acquire_interner_lock();
    guard_error_stats::with_test_lock(|| {
        // Start from a known baseline. Use unchecked variant because
        // populate_reset_targets calls begin_model_check_run() which would
        // cause the guarded reset_global_state() to skip clearing.
        reset_global_state_unchecked();

        let (snapshots, model_check_run) = populate_reset_targets(next_reset_test_id());

        // Balance the begin_model_check_run() from populate_reset_targets
        // before the unchecked reset. This ensures the counter is 0 (or
        // close to it) after the reset, matching the expected post-reset state
        // where begin_model_check_run() triggers clearing on the 0-to-1 transition.
        drop(model_check_run);

        // Reset everything. The counter should be 0 after end_model_check_run.
        reset_global_state_unchecked();

        assert_all_tables_cleared_after_reset(&snapshots);

        // Regression: tla-eval formerly cached the process-global NameId for
        // EXCEPT's special `@` binding in a LazyLock.  A run reset invalidated
        // that ID, so a later spec silently bound the old value under an
        // unrelated name (CoffeeCan strict certification then failed closed).
        // Prime the binding, record its current slot, reset the interner, and
        // deliberately occupy that slot with dummy names before evaluating a
        // freshly lowered EXCEPT expression.
        reset_global_state_unchecked();
        assert_eq!(eval_except_at_binding(), Value::int(2));
        let old_at_id = tla_core::lookup_name_id("@").expect("@ must be interned");
        reset_global_state_unchecked();
        for index in 0..=old_at_id.0 {
            let _ = tla_core::intern_name(&format!("reset_except_dummy_{index}"));
        }
        assert_eq!(
            eval_except_at_binding(),
            Value::int(2),
            "EXCEPT @ must re-intern its binding after the name interner resets"
        );
        reset_global_state_unchecked();
    });
}
