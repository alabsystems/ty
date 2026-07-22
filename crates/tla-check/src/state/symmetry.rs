// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Symmetry reduction operations for `State`.
//!
//! Extracted from `state.rs` (#3607). Contains permutation application and
//! canonical fingerprinting under symmetry groups.

use std::cmp::Ordering;
use std::sync::Arc;

#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use crate::value::{FuncValue, MVPerm};
use crate::Value;
use tla_core::kani_types::OrdMap;

use super::value_hash::compute_fingerprint_from_min_vals;
use super::{Fingerprint, State};

// Profiling counters for symmetry fingerprinting — debug builds only
#[cfg(debug_assertions)]
static SYMMETRY_FP_CALLS: AtomicU64 = AtomicU64::new(0);
#[cfg(debug_assertions)]
static SYMMETRY_FP_US: AtomicU64 = AtomicU64::new(0);

/// Print and reset symmetry fingerprinting statistics.
/// In release builds, this is a no-op (counters don't exist).
#[cfg(debug_assertions)]
pub(crate) fn print_symmetry_stats() {
    let calls = SYMMETRY_FP_CALLS.swap(0, AtomicOrdering::Relaxed);
    let us = SYMMETRY_FP_US.swap(0, AtomicOrdering::Relaxed);
    if calls > 0 {
        eprintln!(
            "=== Symmetry Fingerprint Profile ===\n  Calls: {}\n  Time: {:.3}s\n  Avg: {:.1}µs/call",
            calls,
            us as f64 / 1_000_000.0,
            us as f64 / calls as f64
        );
    }
}
#[cfg(not(debug_assertions))]
pub(crate) fn print_symmetry_stats() {}

impl State {
    /// Compute the canonical fingerprint under symmetry permutations
    ///
    /// For symmetry reduction, we need to identify symmetric states as equivalent.
    /// This is done by finding the lexicographically smallest permuted state
    /// and returning its fingerprint (TLC-compatible algorithm).
    ///
    /// IMPORTANT: We find lexmin(S, P1(S), P2(S), ...) then fingerprint it.
    /// NOT min(fp(S), fp(P1(S)), ...) - fingerprint order != lexicographic order!
    ///
    /// If `perms` is empty, returns the regular fingerprint.
    /// Results are cached for efficiency when called multiple times on the same state.
    ///
    /// # Algorithm
    ///
    /// Lazy permute-compare (sharpens TLC's interleaved pattern,
    /// TLCStateMut.java:212-247):
    /// - Phase 1 compares each variable's PERMUTED value against the current
    ///   minimum via `Value::permute_cmp` — streaming through the value
    ///   structure with NO materialization of the permuted state. Most
    ///   permutations lose (`Greater`) at the first differing variable and
    ///   cost zero allocations.
    /// - The running minimum itself is lazy: it is `(best_perm, original
    ///   state)` with per-variable slots that materialize as exactly
    ///   `permute(original_i, best_perm)` the first time a later comparison
    ///   (or the final fingerprint) needs them. A new minimum (strictly
    ///   Less) just retargets `best_perm` and clears the slots — variables
    ///   of intermediate minimums that are never compared again are never
    ///   built.
    /// - Permutations that tie the minimum exactly keep the incumbent (the
    ///   eager algorithm also only swapped on strictly-Less).
    ///
    /// `permute_cmp` is bit-exact with `permute(...).cmp(...)` (fail-closed
    /// materialization for shapes it cannot stream), and every materialized
    /// minimum slot is the same `permute(original, best_perm)` expression the
    /// eager algorithm stored, so the selected canonical representative — and
    /// therefore the fingerprint — is unchanged.
    pub fn fingerprint_with_symmetry(&self, perms: &[FuncValue]) -> Fingerprint {
        if perms.is_empty() {
            return self.fingerprint();
        }

        // Return cached value if available
        if let Some(&cached) = self.canonical_fingerprint.get() {
            return cached;
        }

        #[cfg(debug_assertions)]
        let start = std::time::Instant::now();
        #[cfg(debug_assertions)]
        SYMMETRY_FP_CALLS.fetch_add(1, AtomicOrdering::Relaxed);

        // Permuted values are throwaways (compared, possibly fingerprinted,
        // then dropped) — skip intern-table traffic while building them.
        let _intern_guard = tla_value::value::InterningSkipGuard::new();

        // Convert to Vec for indexed access (OrdMap iterates in sorted order)
        let vars_vec: Vec<(&Arc<str>, &Value)> = self.vars.iter().collect();

        // The current lexicographic minimum is `best_perm` applied to this
        // state (`None` = the state itself). Entries materialize on demand:
        // a `None` slot stands for `permute(original, best_perm)` and is
        // filled exactly with that expression when first compared — so every
        // materialized slot is bit-identical to what the eager algorithm
        // would have stored, and slots never compared again are never built.
        let mut best_perm: Option<&FuncValue> = None;
        let mut min_vals: Vec<Option<Value>> =
            vars_vec.iter().map(|(_, v)| Some((*v).clone())).collect();

        'next_perm: for perm in perms {
            // Phase 1: lazy streaming compare, no permuted-state
            // materialization.
            let mut strictly_less = false;
            for (i, (_, value)) in vars_vec.iter().enumerate() {
                let min_val = min_vals[i].get_or_insert_with(|| {
                    value.permute(best_perm.expect("pending min slot requires a best perm"))
                });
                match value.permute_cmp(perm, min_val) {
                    Ordering::Greater => continue 'next_perm,
                    Ordering::Less => {
                        strictly_less = true;
                        break;
                    }
                    Ordering::Equal => {}
                }
            }
            if !strictly_less {
                // Permuted state ties the current minimum — keep incumbent.
                continue;
            }

            // Phase 2: new minimum — just retarget; entries materialize
            // lazily on the next compare that needs them.
            best_perm = Some(perm);
            for slot in &mut min_vals {
                *slot = None;
            }
        }

        // Finalize: materialize the remaining slots of the minimum.
        let min_vals: Vec<Value> = vars_vec
            .iter()
            .zip(min_vals)
            .map(|((_, value), slot)| {
                slot.unwrap_or_else(|| {
                    value.permute(best_perm.expect("pending min slot requires a best perm"))
                })
            })
            .collect();

        // Compute fingerprint from the minimum values
        // Uses same algorithm as compute_fingerprint but with our Vec
        let canonical_fp = compute_fingerprint_from_min_vals(&vars_vec, &min_vals);

        // Cache the result (ignore if another thread beat us to it)
        let _ = self.canonical_fingerprint.set(canonical_fp);

        #[cfg(debug_assertions)]
        SYMMETRY_FP_US.fetch_add(start.elapsed().as_micros() as u64, AtomicOrdering::Relaxed);
        canonical_fp
    }

    /// Compute the canonical fingerprint using MVPerm for O(1) lookups (Part of #358).
    ///
    /// This is 10x faster than `fingerprint_with_symmetry()` for specs with many
    /// model values because MVPerm uses array indexing instead of binary search.
    pub fn fingerprint_with_symmetry_fast(&self, mvperms: &[MVPerm]) -> Fingerprint {
        if mvperms.is_empty() {
            return self.fingerprint();
        }

        // Return cached value if available
        if let Some(&cached) = self.canonical_fingerprint.get() {
            return cached;
        }

        #[cfg(debug_assertions)]
        let start = std::time::Instant::now();
        #[cfg(debug_assertions)]
        SYMMETRY_FP_CALLS.fetch_add(1, AtomicOrdering::Relaxed);

        // Permuted values are throwaways (compared, possibly fingerprinted,
        // then dropped) — skip intern-table traffic while building them.
        let _intern_guard = tla_value::value::InterningSkipGuard::new();

        // Convert to Vec for indexed access (OrdMap iterates in sorted order)
        let vars_vec: Vec<(&Arc<str>, &Value)> = self.vars.iter().collect();

        // Lazily-materialized minimum — see `fingerprint_with_symmetry`
        // for the algorithm notes.
        let mut best_perm: Option<&MVPerm> = None;
        let mut min_vals: Vec<Option<Value>> =
            vars_vec.iter().map(|(_, v)| Some((*v).clone())).collect();

        'next_perm: for mvperm in mvperms {
            // Phase 1: lazy streaming compare, no permuted-state
            // materialization.
            let mut strictly_less = false;
            for (i, (_, value)) in vars_vec.iter().enumerate() {
                let min_val = min_vals[i].get_or_insert_with(|| {
                    value.permute_fast(best_perm.expect("pending min slot requires a best perm"))
                });
                // O(1) permutation via MVPerm instead of O(log n) binary search
                match value.permute_cmp_fast(mvperm, min_val) {
                    Ordering::Greater => continue 'next_perm,
                    Ordering::Less => {
                        strictly_less = true;
                        break;
                    }
                    Ordering::Equal => {}
                }
            }
            if !strictly_less {
                // Permuted state ties the current minimum — keep incumbent.
                continue;
            }

            // Phase 2: new minimum — just retarget; entries materialize
            // lazily on the next compare that needs them.
            best_perm = Some(mvperm);
            for slot in &mut min_vals {
                *slot = None;
            }
        }

        // Finalize: materialize the remaining slots of the minimum.
        let min_vals: Vec<Value> = vars_vec
            .iter()
            .zip(min_vals)
            .map(|((_, value), slot)| {
                slot.unwrap_or_else(|| {
                    value.permute_fast(best_perm.expect("pending min slot requires a best perm"))
                })
            })
            .collect();

        // Compute fingerprint from the minimum values
        // Uses same algorithm as compute_fingerprint but with our Vec
        let canonical_fp = compute_fingerprint_from_min_vals(&vars_vec, &min_vals);

        // Cache the result (ignore if another thread beat us to it)
        let _ = self.canonical_fingerprint.set(canonical_fp);

        #[cfg(debug_assertions)]
        SYMMETRY_FP_US.fetch_add(start.elapsed().as_micros() as u64, AtomicOrdering::Relaxed);
        canonical_fp
    }

    /// Apply a permutation to all values in this state
    ///
    /// Returns a new state with all model values permuted according to the given
    /// permutation function. Used for symmetry reduction.
    pub fn permute(&self, perm: &FuncValue) -> State {
        let permuted_vars: OrdMap<Arc<str>, Value> = self
            .vars
            .iter()
            .map(|(name, value)| (name.clone(), value.permute(perm)))
            .collect();
        State::from_vars(permuted_vars)
    }

    /// Apply a permutation using MVPerm for O(1) lookups (Part of #358).
    pub fn permute_fast(&self, mvperm: &MVPerm) -> State {
        let permuted_vars: OrdMap<Arc<str>, Value> = self
            .vars
            .iter()
            .map(|(name, value)| (name.clone(), value.permute_fast(mvperm)))
            .collect();
        State::from_vars(permuted_vars)
    }
}
