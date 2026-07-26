// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Hot-path lookup/insert helpers for sets, int-functions, strings, and TLC
//! string tokens during parallel BFS.
//!
//! Part of #3412: extracted from `parallel_intern.rs` (lines 364-610).

use super::super::Value;
use super::state::{worker_overlay_cap, WorkerInternState, WORKER_INTERN, WORKER_TOKEN_COUNTER};
use crate::rp::Rp as Arc;
use std::sync::atomic::Ordering;

/// Sound cliff-clear of the per-worker `set_overlay` when its growth beyond the
/// preloaded frozen baseline would exceed `worker_overlay_cap()`.
///
/// SOUNDNESS: the `set_overlay` "token" is the `Arc` pointer, which never feeds
/// `Value::eq`/`Value::cmp` (used only as a `ptr_eq` fast path that falls through
/// to content) or state fingerprints (which hash element content). Dropping the
/// overlay and re-minting a fresh `Arc` for an evicted set on a later miss yields
/// an equal-comparing, identically-fingerprinting value. Lookups for entries that
/// were preloaded from the frozen snapshot still hit `state.frozen` after the
/// clear, so no value becomes un-interned in a way that changes a verdict.
#[inline]
pub(super) fn maybe_evict_set_overlay(state: &mut WorkerInternState, cap: usize) {
    if cap != usize::MAX
        && state
            .set_overlay
            .len()
            .saturating_sub(state.set_overlay_base)
            >= cap
    {
        state.set_overlay.clear();
        state.set_overlay.shrink_to_fit();
        state.set_overlay_base = 0;
    }
}

/// Sound cliff-clear of the per-worker `int_func_overlay`. Same soundness
/// argument as [`maybe_evict_set_overlay`]: the `Arc` is dedup-only and never
/// participates in equality, ordering, or fingerprinting.
#[inline]
pub(super) fn maybe_evict_int_func_overlay(state: &mut WorkerInternState, cap: usize) {
    if cap != usize::MAX
        && state
            .int_func_overlay
            .len()
            .saturating_sub(state.int_func_overlay_base)
            >= cap
    {
        state.int_func_overlay.clear();
        state.int_func_overlay.shrink_to_fit();
        state.int_func_overlay_base = 0;
    }
}

/// Try to find or insert a set in the parallel intern path.
///
/// Takes `elements` by reference for lookups; only allocates a new `Arc` on
/// cache miss (uncommon during BFS since most values are in the frozen snapshot).
///
/// Returns `Some(arc)` if parallel mode is active (hit or inserted into overlay).
/// Returns `None` if the worker intern scope is not installed.
#[inline]
pub(in crate::value) fn parallel_intern_set(fp: u64, elements: &[Value]) -> Option<Arc<[Value]>> {
    WORKER_INTERN.with(|c| {
        // Shared borrow for lookups
        {
            let state = c.borrow();
            let state = state.as_ref()?;

            // Check worker-local overlay first (most recently created values)
            if let Some(arc) = state.set_overlay.get(&fp) {
                if arc.len() == elements.len()
                    && arc.iter().zip(elements.iter()).all(|(a, b)| a == b)
                {
                    state.overlay_set_hits.set(state.overlay_set_hits.get() + 1);
                    return Some(Arc::clone(arc));
                }
            }

            // Check frozen snapshot
            if let Some(arc) = state.frozen.sets.get(&fp) {
                if arc.len() == elements.len()
                    && arc.iter().zip(elements.iter()).all(|(a, b)| a == b)
                {
                    state.frozen_set_hits.set(state.frozen_set_hits.get() + 1);
                    return Some(Arc::clone(arc));
                }
            }
        }
        // Shared borrow dropped — safe to mutably borrow for insert

        let arc: Arc<[Value]> = Arc::from(elements.to_vec());
        if let Some(state) = c.borrow_mut().as_mut() {
            state.new_set_inserts.set(state.new_set_inserts.get() + 1);
            // Bound per-worker memory: cliff-clear (sound — Arc is dedup-only)
            // before inserting if growth beyond the frozen baseline hits the cap.
            maybe_evict_set_overlay(state, worker_overlay_cap());
            state.set_overlay.insert(fp, Arc::clone(&arc));
        }
        Some(arc)
    })
}

/// Try to find an interned int-function modification in the parallel intern path.
///
/// Returns `Some(Some(arc))` if found in overlay or frozen snapshot.
/// Returns `Some(None)` if in parallel mode but not found.
/// Returns `None` if the worker intern scope is not installed.
#[inline]
#[allow(clippy::option_option)] // 3-state: None=not active, Some(None)=miss, Some(Some)=hit
pub(in crate::value) fn parallel_try_get_interned_modified(
    fp: u64,
    values: &[Value],
    arr_idx: usize,
    new_value: &Value,
) -> Option<Option<Arc<Vec<Value>>>> {
    WORKER_INTERN.with(|c| {
        let state = c.borrow();
        let state = state.as_ref()?;

        // Check overlay
        if let Some(arc) = state.int_func_overlay.get(&fp) {
            if arc.len() == values.len() {
                let matches = arc.iter().enumerate().all(|(i, v)| {
                    if i == arr_idx {
                        v == new_value
                    } else {
                        v == &values[i]
                    }
                });
                if matches {
                    state
                        .overlay_int_func_hits
                        .set(state.overlay_int_func_hits.get() + 1);
                    return Some(Some(Arc::clone(arc)));
                }
            }
        }

        // Check frozen
        if let Some(arc) = state.frozen.int_funcs.get(&fp) {
            if arc.len() == values.len() {
                let matches = arc.iter().enumerate().all(|(i, v)| {
                    if i == arr_idx {
                        v == new_value
                    } else {
                        v == &values[i]
                    }
                });
                if matches {
                    state
                        .frozen_int_func_hits
                        .set(state.frozen_int_func_hits.get() + 1);
                    return Some(Some(Arc::clone(arc)));
                }
            }
        }

        Some(None) // In parallel mode but not found
    })
}

/// Try to find or insert an int-function in the parallel intern path.
///
/// Takes `values` by reference for lookups; only allocates on cache miss.
///
/// Returns `Some(arc)` if parallel mode is active (hit or inserted into overlay).
/// Returns `None` if the worker intern scope is not installed.
#[inline]
pub(in crate::value) fn parallel_intern_int_func(
    fp: u64,
    values: &[Value],
) -> Option<Arc<Vec<Value>>> {
    WORKER_INTERN.with(|c| {
        // Shared borrow for lookups
        {
            let state = c.borrow();
            let state = state.as_ref()?;

            // Check overlay
            if let Some(arc) = state.int_func_overlay.get(&fp) {
                if arc.len() == values.len() && arc.iter().zip(values.iter()).all(|(a, b)| a == b) {
                    state
                        .overlay_int_func_hits
                        .set(state.overlay_int_func_hits.get() + 1);
                    return Some(Arc::clone(arc));
                }
            }

            // Check frozen
            if let Some(arc) = state.frozen.int_funcs.get(&fp) {
                if arc.len() == values.len() && arc.iter().zip(values.iter()).all(|(a, b)| a == b) {
                    state
                        .frozen_int_func_hits
                        .set(state.frozen_int_func_hits.get() + 1);
                    return Some(Arc::clone(arc));
                }
            }
        }

        let arc = Arc::new(values.to_vec());
        if let Some(state) = c.borrow_mut().as_mut() {
            state
                .new_int_func_inserts
                .set(state.new_int_func_inserts.get() + 1);
            // Bound per-worker memory: cliff-clear (sound — Arc is dedup-only)
            // before inserting if growth beyond the frozen baseline hits the cap.
            maybe_evict_int_func_overlay(state, worker_overlay_cap());
            state.int_func_overlay.insert(fp, Arc::clone(&arc));
        }
        Some(arc)
    })
}

// ============================================================================
// Hot-path lookup/insert functions for string interning (Part of #3285)
// ============================================================================

/// Try to find or insert a string in the parallel intern path.
///
/// Returns `Some(arc)` if parallel mode is active (hit or inserted into overlay).
/// Returns `None` if the worker intern scope is not installed.
#[inline]
pub(in crate::value) fn parallel_intern_string(s: &str) -> Option<Arc<str>> {
    WORKER_INTERN.with(|c| {
        // Shared borrow for lookups
        {
            let state = c.borrow();
            let state = state.as_ref()?;

            // Check worker-local overlay first
            if let Some(arc) = state.string_overlay.get(s) {
                state
                    .overlay_string_hits
                    .set(state.overlay_string_hits.get() + 1);
                return Some(Arc::clone(arc));
            }

            // Check frozen snapshot
            if let Some(arc) = state.frozen.strings.get(s) {
                state
                    .frozen_string_hits
                    .set(state.frozen_string_hits.get() + 1);
                return Some(Arc::clone(arc));
            }
        }
        // Shared borrow dropped — safe to mutably borrow for insert

        let arc: Arc<str> = Arc::from(s);
        if let Some(state) = c.borrow_mut().as_mut() {
            state
                .new_string_inserts
                .set(state.new_string_inserts.get() + 1);
            state.string_overlay.insert(s.to_string(), Arc::clone(&arc));
            // Part of #3287: Eagerly assign TLC string token at intern time.
            // This string is new (not in frozen or overlay), so assign a fresh
            // token now rather than deferring to comparison time. Inlined here
            // because we're already inside WORKER_INTERN.with() and can't
            // re-enter it via parallel_tlc_string_token().
            let tok = WORKER_TOKEN_COUNTER.fetch_add(1, Ordering::Relaxed);
            state.string_token_overlay.insert(Arc::clone(&arc), tok);
        }
        Some(arc)
    })
}

/// Try to get a TLC string token from the parallel intern path.
///
/// Returns `Some(token)` if parallel mode is active (hit or assigned new token).
/// Returns `None` if the worker intern scope is not installed.
#[inline]
pub(in crate::value) fn parallel_tlc_string_token(s: &Arc<str>) -> Option<u32> {
    WORKER_INTERN.with(|c| {
        // Shared borrow for lookups
        {
            let state = c.borrow();
            let state = state.as_ref()?;

            // Check worker-local overlay first
            if let Some(&tok) = state.string_token_overlay.get(s.as_ref()) {
                state
                    .overlay_token_hits
                    .set(state.overlay_token_hits.get() + 1);
                return Some(tok);
            }

            // Check frozen snapshot
            if let Some(&tok) = state.frozen.string_tokens.get(s.as_ref()) {
                state
                    .frozen_token_hits
                    .set(state.frozen_token_hits.get() + 1);
                return Some(tok);
            }
        }
        // Shared borrow dropped — safe to mutably borrow for insert

        // Assign a new token from the shared atomic counter.
        // Workers may assign tokens in different orders, but TLC token ordering
        // only needs to be consistent within a single run's comparison operations.
        let tok = WORKER_TOKEN_COUNTER.fetch_add(1, Ordering::Relaxed);
        if let Some(state) = c.borrow_mut().as_mut() {
            state.string_token_overlay.insert(Arc::clone(s), tok);
        }
        Some(tok)
    })
}
