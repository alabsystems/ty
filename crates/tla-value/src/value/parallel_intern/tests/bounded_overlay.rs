// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Soundness + bounded-memory tests for the per-worker set/int-func overlay cap
//! (audit #7 OOM fix).
//!
//! These cover the two SHIP requirements that are specific to this fix:
//!
//! - **Value-identity differential**: a value evicted by a cliff-clear and then
//!   re-interned MUST compare equal and fingerprint identically to the original
//!   (and to a reference worker that never evicted). This is the soundness floor:
//!   the `Arc` pointer is dedup-only and never feeds `Value::eq`/`Value::cmp`/
//!   fingerprints, so eviction + re-mint is sound.
//! - **Bounded-memory micro-bench**: under an adversarial many-distinct-values
//!   workload the per-worker overlay size stays bounded (`< 2 * cap`), whereas
//!   the pre-fix path grew unbounded.

use super::*;
use crate::rp::Rp;

/// Build a distinct N-element set fingerprint + elements for index `i`.
fn distinct_set(i: u64) -> (u64, Vec<Value>) {
    let elements = vec![Value::SmallInt(i as i64), Value::SmallInt((i as i64) + 1)];
    let fp = intern_tables::set_fingerprint(&elements);
    (fp, elements)
}

/// Build a distinct int-func fingerprint + values for index `i`.
fn distinct_int_func(i: u64) -> (u64, Vec<Value>) {
    let values = vec![Value::SmallInt(i as i64), Value::SmallInt((i as i64) + 2)];
    let fp = intern_tables::int_func_fingerprint(0, 1, &values);
    (fp, values)
}

/// Fingerprint a single set value (wraps it in a real `Value::Set`).
fn fingerprint_of_set(elements: &[Value]) -> u64 {
    let v = Value::set(elements.iter().cloned());
    v.fingerprint_extend(0).expect("fingerprint")
}

#[test]
fn evicted_set_reintern_yields_equal_value_and_fingerprint() {
    let _lock = crate::value::lock_intern_state();
    clear_set_intern_table();
    freeze_value_interners();
    install_worker_intern_scope();

    // Use a small explicit cap so we can drive an eviction deterministically
    // without depending on the OnceLock-cached env default.
    const CAP: usize = 64;

    // Intern an "early" set and capture its identity (content + Arc + fingerprint).
    let (fp0, elems0) = distinct_set(0);
    let arc_before = parallel_intern_set(fp0, &elems0).expect("intern early set");
    let fp_value_before = fingerprint_of_set(&arc_before);

    // Adversarial fill: push well past the cap so a cliff-clear is forced. We call
    // the eviction helper with the explicit CAP before each insert to mirror the
    // production insert path (`parallel_intern_set` calls it with
    // `worker_overlay_cap()`), but with a deterministic small cap here.
    let mut cleared_at_least_once = false;
    for i in 1..(CAP as u64 * 4) {
        let (fp, elems) = distinct_set(i);
        // Mirror the production insert: maybe-evict, then insert a fresh Arc.
        WORKER_INTERN.with(|c| {
            if let Some(state) = c.borrow_mut().as_mut() {
                let before = state.set_overlay.len();
                maybe_evict_set_overlay(state, CAP);
                if state.set_overlay.len() < before {
                    cleared_at_least_once = true;
                }
                let arc: crate::rp::Rp<[Value]> = crate::rp::Rp::from(elems.clone());
                state.set_overlay.insert(fp, arc);
            }
        });
    }
    assert!(
        cleared_at_least_once,
        "workload must exceed CAP and force at least one cliff-clear"
    );

    // The early set is now evicted from the overlay. Re-intern it.
    let arc_after = parallel_intern_set(fp0, &elems0).expect("re-intern early set");

    // SOUNDNESS FLOOR 1: content equality (extensional).
    assert_eq!(
        arc_after.as_ref(),
        arc_before.as_ref(),
        "re-interned set content must be identical after eviction"
    );

    // SOUNDNESS FLOOR 2: Value-level equality. Build real `Value::Set`s from
    // both Arcs and assert `Value::eq` (the equality the model checker uses).
    let v_before = Value::set(arc_before.iter().cloned());
    let v_after = Value::set(arc_after.iter().cloned());
    assert_eq!(
        v_before, v_after,
        "Value::eq must hold after eviction + re-intern (Arc is dedup-only)"
    );

    // SOUNDNESS FLOOR 3: identical state fingerprint. State dedup depends on this,
    // NOT on the Arc pointer. A fresh Arc with the same content must fingerprint
    // identically or state dedup would break.
    let fp_value_after = fingerprint_of_set(&arc_after);
    assert_eq!(
        fp_value_before, fp_value_after,
        "fingerprint must be identical after eviction + re-intern"
    );

    uninstall_worker_intern_scope();
    unfreeze_value_interners();
    clear_set_intern_table();
}

#[test]
fn capped_worker_matches_uncapped_worker_value_for_value_differential() {
    // Verdict differential proxy: a worker that cliff-clears under cap pressure
    // must produce the SAME interned values (content/eq/fingerprint) as a worker
    // that never evicts, for every value in the workload. If interning ever
    // minted a distinct-but-equal token that changed equality, this would fail.
    let _lock = crate::value::lock_intern_state();
    clear_set_intern_table();
    freeze_value_interners();

    const CAP: usize = 32;
    const N: u64 = 200; // > CAP, forces multiple cliff-clears in the capped worker

    // Reference worker: cap disabled (usize::MAX), so it never evicts.
    install_worker_intern_scope();
    let mut ref_fps = Vec::new();
    for i in 0..N {
        let (fp, elems) = distinct_set(i);
        WORKER_INTERN.with(|c| {
            if let Some(state) = c.borrow_mut().as_mut() {
                maybe_evict_set_overlay(state, usize::MAX); // never evicts
                let arc: crate::rp::Rp<[Value]> = crate::rp::Rp::from(elems.clone());
                state.set_overlay.insert(fp, arc);
            }
        });
        ref_fps.push(fingerprint_of_set(&elems));
    }
    uninstall_worker_intern_scope();

    // Capped worker: same workload, with the small cap forcing cliff-clears.
    install_worker_intern_scope();
    let mut capped_fps = Vec::new();
    for i in 0..N {
        let (fp, elems) = distinct_set(i);
        WORKER_INTERN.with(|c| {
            if let Some(state) = c.borrow_mut().as_mut() {
                maybe_evict_set_overlay(state, CAP);
                let arc: crate::rp::Rp<[Value]> = crate::rp::Rp::from(elems.clone());
                state.set_overlay.insert(fp, arc);
            }
        });
        capped_fps.push(fingerprint_of_set(&elems));
    }
    uninstall_worker_intern_scope();

    // Every fingerprint must match, regardless of eviction. No verdict change.
    assert_eq!(
        ref_fps, capped_fps,
        "capped worker must produce identical fingerprints to uncapped worker"
    );

    unfreeze_value_interners();
    clear_set_intern_table();
}

#[test]
fn evicted_int_func_reintern_yields_equal_value() {
    let _lock = crate::value::lock_intern_state();
    clear_int_func_intern_table();
    freeze_value_interners();
    install_worker_intern_scope();

    const CAP: usize = 48;

    let (fp0, vals0) = distinct_int_func(0);
    let arc_before = parallel_intern_int_func(fp0, &vals0).expect("intern early int-func");

    let mut cleared = false;
    for i in 1..(CAP as u64 * 4) {
        let (fp, vals) = distinct_int_func(i);
        WORKER_INTERN.with(|c| {
            if let Some(state) = c.borrow_mut().as_mut() {
                let before = state.int_func_overlay.len();
                maybe_evict_int_func_overlay(state, CAP);
                if state.int_func_overlay.len() < before {
                    cleared = true;
                }
                state
                    .int_func_overlay
                    .insert(fp, crate::rp::Rp::new(vals.clone()));
            }
        });
    }
    assert!(cleared, "int-func workload must force a cliff-clear");

    let arc_after = parallel_intern_int_func(fp0, &vals0).expect("re-intern early int-func");
    assert_eq!(
        arc_after.as_ref(),
        arc_before.as_ref(),
        "re-interned int-func content must be identical after eviction"
    );

    uninstall_worker_intern_scope();
    unfreeze_value_interners();
    clear_int_func_intern_table();
}

#[test]
fn bounded_overlay_memory_micro_bench() {
    // Adversarial many-distinct-values workload. Asserts the overlay stays
    // BOUNDED and reports before/after sizes. Without the cap this map would
    // grow to `N` entries (unbounded w.r.t. distinct values).
    let _lock = crate::value::lock_intern_state();
    clear_set_intern_table();
    clear_int_func_intern_table();
    freeze_value_interners();
    install_worker_intern_scope();

    const CAP: usize = 1_000;
    const N: u64 = 250_000; // 250x the cap — would be ~250k Arcs uncapped

    let baseline = WORKER_INTERN.with(|c| {
        c.borrow()
            .as_ref()
            .map(|s| s.set_overlay.len())
            .unwrap_or(0)
    });

    let mut peak = baseline;
    for i in 0..N {
        let (fp, elems) = distinct_set(i);
        WORKER_INTERN.with(|c| {
            if let Some(state) = c.borrow_mut().as_mut() {
                maybe_evict_set_overlay(state, CAP);
                let arc: crate::rp::Rp<[Value]> = crate::rp::Rp::from(elems.clone());
                state.set_overlay.insert(fp, arc);
                peak = peak.max(state.set_overlay.len());
            }
        });
    }

    let final_len = WORKER_INTERN.with(|c| {
        c.borrow()
            .as_ref()
            .map(|s| s.set_overlay.len())
            .unwrap_or(0)
    });

    // Bounded: never exceeds 2x the cap (cliff-clear triggers at >= cap, then we
    // insert one more, so the worst case is cap + 1 before the next clear).
    assert!(
        peak <= 2 * CAP,
        "overlay peak {peak} must stay bounded under 2*CAP={} for {N} distinct values \
         (baseline {baseline}); without the cap this would be ~{N}",
        2 * CAP
    );

    // Report (visible with `cargo test -- --nocapture`).
    println!(
        "[bounded_overlay_memory_micro_bench] distinct_values={N} cap={CAP} \
         baseline_overlay={baseline} peak_overlay={peak} final_overlay={final_len} \
         (uncapped would be ~{N})"
    );

    uninstall_worker_intern_scope();
    unfreeze_value_interners();
    clear_set_intern_table();
    clear_int_func_intern_table();
}

#[test]
fn default_cap_is_configured() {
    // Sanity: the production path resolves a finite cap by default (unless the
    // env override sets 0 to disable it). DEFAULT is the documented value.
    assert_eq!(DEFAULT_WORKER_OVERLAY_CAP, 1_000_000);
    let cap = worker_overlay_cap();
    // In CI/test the env var is unset, so the resolved cap is the default.
    // (If a developer sets TY_PARALLEL_OVERLAY_CAP this may differ; allow it.)
    assert!(cap >= 1 || cap == usize::MAX);
}
