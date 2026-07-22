// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Per-parent seeding of implied-action zero-arg external values.
//!
//! During implied-action checking, one parent's successor batch re-evaluates
//! the term once per edge, and each evaluation probes the PARENT-side pinned
//! refinement operators (`token` / `pending`, unprimed) through the
//! fingerprint-keyed transition memo — the values are identical across the
//! whole batch because the parent binding does not change. This module hoists
//! those probes to once per (parent, term): the values are collected from
//! VALIDATED transition-memo hits and handed to the bytecode VM as
//! `CallExternal` seeds, so per-edge executions skip the memo machinery
//! entirely for parent-side references.
//!
//! # Soundness
//!
//! * Value provenance: seeds come exclusively from
//!   `implied_parent_side_external_probe` — a validated transition-memo hit.
//!   Store-side eligibility (`transition_deps_eligible`) guarantees the
//!   entry's value depends ONLY on parent-side state-variable reads, and hit
//!   validation compared every recorded dep against the CURRENTLY BOUND
//!   parent array. The value is therefore exactly what
//!   `eval_zero_arg_external(ctx, name, prime=false)` returns under this
//!   parent binding.
//! * Reuse window: a seed set is keyed by the parent's values-allocation
//!   pointer AND a held `ArrayState` clone (keepalive). Holding the clone
//!   pins the allocation (no ABA reuse), and `ArrayState` mutation is
//!   copy-on-write (`Arc::make_mut`), so pointer equality proves the bound
//!   parent values are bit-identical to the ones the probes validated
//!   against.
//! * Mode: seeds are built and consumed only in `Current` lookup mode (both
//!   sides gate on `state_lookup_mode_is_current`), matching the mode the
//!   probe's value was validated for.
//! * Fail open: any probe miss simply omits that seed — the VM falls back to
//!   the full `eval_zero_arg_external` path for that operator, byte-identical
//!   to a run without seeding.
//!
//! Kill switch: `TY_NO_IMPLIED_EXTERNAL_SEEDS=1` (no seeds are ever built or
//! attached).

use smallvec::SmallVec;
use tla_value::Rp;
use std::cell::RefCell;
use std::sync::Arc;

use super::implied_verdict_cache::ImpliedVerdictCacheSpec;
use crate::eval::EvalCtx;
use crate::state::ArrayState;
use tla_value::Value;

feature_flag!(
    pub(crate) no_implied_external_seeds,
    "TY_NO_IMPLIED_EXTERNAL_SEEDS"
);

/// Seed tuples in the VM's `CallExternal` memo shape: (name value, prime
/// mode, result value). Prime mode is always `false` (parent side).
pub(crate) type SeedVec = SmallVec<[(Value, bool, Value); 4]>;

struct SeedCache {
    /// Keepalive pinning the parent's values allocation (see module docs).
    keepalive: Option<ArrayState>,
    /// `parent.values().as_ptr()` of the keepalive.
    parent_values_ptr: usize,
    /// Per-term seed sets for this parent.
    per_term: SmallVec<[(u64, SeedVec); 2]>,
}

thread_local! {
    static SEED_CACHE: RefCell<SeedCache> = RefCell::new(SeedCache {
        keepalive: None,
        parent_values_ptr: 0,
        per_term: SmallVec::new(),
    });
}

/// Fetch (or build) the parent-side external seeds for `spec`'s term under
/// the currently bound transition. Returns an empty vec when seeding is
/// disabled, the mode is not `Current`, or no probe validated.
pub(crate) fn parent_external_seeds(
    ctx: &EvalCtx,
    spec: &ImpliedVerdictCacheSpec,
    parent: &ArrayState,
) -> SeedVec {
    if no_implied_external_seeds()
        || spec.zero_arg_externals.is_empty()
        || !tla_eval::state_lookup_mode_is_current(ctx)
    {
        return SeedVec::new();
    }
    let ptr = parent.values().as_ptr() as usize;
    SEED_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        if cache.keepalive.is_some() && cache.parent_values_ptr == ptr {
            if let Some((_, seeds)) = cache
                .per_term
                .iter()
                .find(|(term_id, _)| *term_id == spec.term_id)
            {
                return seeds.clone();
            }
        } else {
            // New parent: reset the window (drop the old keepalive last).
            cache.per_term.clear();
            cache.parent_values_ptr = ptr;
            cache.keepalive = Some(parent.clone());
        }
        let mut seeds = SeedVec::new();
        for name in spec.zero_arg_externals.iter() {
            if let Some(value) = tla_eval::implied_parent_side_external_probe(ctx, name.as_str()) {
                seeds.push((Value::String(Rp::from(name.as_str())), false, value));
            }
        }
        cache.per_term.push((spec.term_id, seeds.clone()));
        seeds
    })
}
