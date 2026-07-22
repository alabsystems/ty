// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Dependency data structures: `DepMap`, `VarDepMap`, `OpEvalDeps`.
//!
//! These track which state variables, next-state variables, and captured
//! locals were read during operator evaluation.

use crate::value::Value;
use crate::var_index::VarIndex;
use smallvec::SmallVec;
use tla_core::name_intern::NameId;
use tla_value::CompactValue;

/// Part of #3025: Inline sorted map for tiny dep sets (typically 1-5 entries).
/// Replaces BTreeMap to avoid per-node heap allocations. Linear scan is faster
/// than tree traversal for n <= ~8. SmallVec<4> stores up to 4 entries inline.
pub(crate) type DepMap<K, V> = SmallVec<[(K, V); 4]>;

/// A captured local dependency: the value the local held when first read, plus
/// the binding depth at which it resolved at *record time*.
///
/// `depth` is the `resolved_stack_index` from `record_local_read` — the index in
/// the global `binding_depth` coordinate at which this local was bound when the
/// read happened. It is the ground-truth basis for the internal-vs-external
/// decision in `strip_internal_locals`: a local whose record-time depth is
/// `>= base_stack_len` was pushed AFTER the operator's evaluation started, so it
/// is an internal iteration variable (quantifier / comprehension / recursive
/// function index) and must not gate cache validity.
///
/// Recording the depth here — rather than re-resolving the NameId against the
/// binding chain at strip time — fixes a conflation bug: when an inner internal
/// variable shadows an identically-named outer binding (e.g. a recursive
/// `F[i \in S] == ... G ... F[i-1]` whose body reaches a CONSTANT operator `G`
/// while an outer `i` is also in scope), the inner `i` is already popped by strip
/// time, so a re-lookup misresolves to the outer `i` (depth `< base`) and wrongly
/// keeps it — marking the constant op inconsistent and defeating its zero-arg
/// cache. The depth is stable in the global coordinate, so it remains valid when
/// the dep propagates up to ancestor frames (which strip against their own,
/// smaller, `base_stack_len`).
///
/// `depth == None` marks a dep whose record-time depth is unknown (reconstructed
/// from a cache that did not retain it); `strip_internal_locals` falls back to the
/// prior binding-chain lookup for those, preserving the established behavior.
#[derive(Clone)]
pub(crate) struct LocalDep {
    pub(crate) value: CompactValue,
    pub(crate) depth: Option<u32>,
}

impl LocalDep {
    #[inline]
    fn new(value: CompactValue, depth: Option<u32>) -> Self {
        LocalDep { value, depth }
    }
}

impl From<CompactValue> for LocalDep {
    #[inline]
    fn from(value: CompactValue) -> Self {
        // Depth-less: used by test fixtures that construct dep sets directly.
        LocalDep { value, depth: None }
    }
}

/// Find existing entry by key in a DepMap. Returns index if found.
#[inline]
fn dep_map_find<K: Eq, V>(map: &DepMap<K, V>, key: &K) -> Option<usize> {
    map.iter().position(|(k, _)| k == key)
}

/// Check if a DepMap contains a given key.
#[inline]
pub(crate) fn dep_map_contains_key<K: Eq, V>(map: &DepMap<K, V>, key: &K) -> bool {
    map.iter().any(|(k, _)| k == key)
}

/// Fix #2414: Dense variable dep map indexed by VarIndex for O(1) record/lookup.
///
/// Replaces `DepMap<VarIndex, Value>` (linear scan per access) with direct
/// array indexing. Typical specs have <20 state variables, so the SmallVec<8>
/// inline capacity covers most cases without heap allocation.
///
/// Part of #3579: Stores `CompactValue` (8B) instead of `Option<Value>` (25B)
/// using `CompactValue::nil()` as the "no dep recorded" sentinel. 3x smaller
/// dep arrays reduce cache pressure on the dep tracking hot path.
///
/// Complexity improvement:
/// - `record_state`/`record_next`: O(n) → O(1)
/// - `merge_from` state/next: O(n*m) → O(n)
#[derive(Clone)]
pub(crate) struct VarDepMap {
    entries: SmallVec<[CompactValue; 8]>,
    count: u16,
}

impl Default for VarDepMap {
    fn default() -> Self {
        Self {
            entries: SmallVec::new(),
            count: 0,
        }
    }
}

impl VarDepMap {
    /// Record a variable read. Returns `true` if the value conflicts with
    /// a previously recorded value for the same variable (inconsistency).
    #[inline]
    pub(crate) fn record(&mut self, idx: VarIndex, value: &Value) -> bool {
        let i = idx.as_usize();
        if i >= self.entries.len() {
            self.entries.resize(i + 1, CompactValue::nil());
        }
        let existing = &self.entries[i];
        if existing.is_nil() {
            self.entries[i] = CompactValue::from(value);
            self.count += 1;
            false
        } else {
            // Part of #3579: compare stored CompactValue against new Value
            // without materializing either side unnecessarily.
            !existing.matches_value(value)
        }
    }

    /// Record from a CompactValue directly (used by merge_from).
    #[inline]
    fn record_compact(&mut self, idx: VarIndex, cv: &CompactValue) -> bool {
        let i = idx.as_usize();
        if i >= self.entries.len() {
            self.entries.resize(i + 1, CompactValue::nil());
        }
        let existing = &self.entries[i];
        if existing.is_nil() {
            self.entries[i] = cv.clone();
            self.count += 1;
            false
        } else {
            existing != cv
        }
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[inline]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn len(&self) -> usize {
        self.count as usize
    }

    /// Check if a variable is recorded.
    #[inline]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn contains_key(&self, idx: &VarIndex) -> bool {
        self.entries
            .get(idx.as_usize())
            .is_some_and(|v| !v.is_nil())
    }

    /// Iterate over recorded (VarIndex, &CompactValue) pairs.
    ///
    /// Part of #3579: yields CompactValue references instead of Value
    /// for zero-alloc comparison on the cache validation hot path.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (VarIndex, &CompactValue)> {
        self.entries.iter().enumerate().filter_map(|(i, cv)| {
            if cv.is_nil() {
                None
            } else {
                Some((VarIndex(i as u16), cv))
            }
        })
    }

    /// Iterate over recorded (VarIndex, &mut CompactValue) pairs. Used by
    /// validate-and-refresh cache probes to swap a validated-equal snapshot
    /// for one sharing the live binding's allocation (see
    /// `transition_entry_valid_refresh`); the map's key set is unchanged.
    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (VarIndex, &mut CompactValue)> {
        self.entries.iter_mut().enumerate().filter_map(|(i, cv)| {
            if cv.is_nil() {
                None
            } else {
                Some((VarIndex(i as u16), cv))
            }
        })
    }

    /// Construct from a slice of (VarIndex, Value) pairs. For test construction.
    #[cfg(test)]
    pub(crate) fn from_entries(entries: &[(VarIndex, Value)]) -> Self {
        let mut map = Self::default();
        for (idx, value) in entries {
            map.record(*idx, value);
        }
        map
    }
}

#[derive(Clone, Default)]
pub(crate) struct OpEvalDeps {
    // Captured locals from the *caller* scope (below base_stack_len).
    // These matter for LET-defined operators that close over bound variables.
    // Part of #3025: SmallVec replaces BTreeMap for 0-3 entry dep sets.
    // Part of #3579: local dep values use CompactValue to avoid cloning full
    // Value payloads on the cache-hit path.
    // Each entry carries the value AND the record-time binding depth (see
    // `LocalDep`), so `strip_internal_locals` can filter internal iteration
    // variables by their captured depth instead of an ambiguous NameId re-lookup.
    pub(crate) local: DepMap<NameId, LocalDep>,
    // Reads of unprimed state variables.
    // Fix #2414: VarDepMap for O(1) lookup instead of O(n) linear scan.
    pub(crate) state: VarDepMap,
    // Reads of primed (next-state) variables, plus unprimed reads while evaluating in next-state mode.
    // Fix #2414: VarDepMap for O(1) lookup instead of O(n) linear scan.
    pub(crate) next: VarDepMap,
    pub(crate) inconsistent: bool,
    // Fix #2991: Track state/next inconsistency separately from local inconsistency.
    pub(crate) state_next_inconsistent: bool,
    // Fix #3062: Track TLCGet("level") dependency for cache invalidation.
    // When an operator reads TLCGet("level"), the BFS level at read time is stored here.
    // Cache entries with a recorded tlc_level are only valid when ctx.tlc_level matches.
    pub(crate) tlc_level: Option<u32>,
    // Fix #3447: Track whether this evaluation touched an INSTANCE LazyBinding.
    // When true, the result may not be stored in a persistent zero-arg/nary partition
    // even if state/next/local/tlc_level are all empty. This prevents stale INSTANCE-
    // mediated operator results from surviving across state boundaries.
    pub(crate) instance_lazy_read: bool,
}

impl OpEvalDeps {
    /// Record a captured local dependency with no known record-time depth.
    ///
    /// Used by paths that reconstruct deps from a cache that did not retain the
    /// depth (e.g. the param-LET cache). `strip_internal_locals` falls back to a
    /// binding-chain lookup for depth-less entries.
    pub(crate) fn record_local(&mut self, name_id: NameId, value: &Value) {
        self.record_local_inner(name_id, CompactValue::from(value), None);
    }

    /// Record a captured local dependency tagged with its record-time binding
    /// depth (the `resolved_stack_index` from `record_local_read`).
    pub(crate) fn record_local_at_depth(&mut self, name_id: NameId, value: &Value, depth: usize) {
        self.record_local_inner(name_id, CompactValue::from(value), Some(depth as u32));
    }

    #[inline]
    fn record_local_inner(&mut self, name_id: NameId, value: CompactValue, depth: Option<u32>) {
        if let Some(idx) = dep_map_find(&self.local, &name_id) {
            if self.local[idx].1.value != value {
                self.inconsistent = true;
            }
            // Prefer a known depth over an unknown one if a later record for the
            // same name carries it; keep the existing known depth otherwise. The
            // depth is a property of the binding, so any known value for this
            // name is consistent.
            if self.local[idx].1.depth.is_none() {
                self.local[idx].1.depth = depth;
            }
        } else {
            self.local.push((name_id, LocalDep::new(value, depth)));
        }
    }

    #[inline]
    fn record_local_compact(&mut self, name_id: NameId, value: &CompactValue, depth: Option<u32>) {
        self.record_local_inner(name_id, value.clone(), depth);
    }

    pub(crate) fn record_state(&mut self, idx: VarIndex, value: &Value) {
        if self.state.record(idx, value) {
            self.inconsistent = true;
            self.state_next_inconsistent = true;
        }
    }

    pub(crate) fn record_next(&mut self, idx: VarIndex, value: &Value) {
        if self.next.record(idx, value) {
            self.inconsistent = true;
            self.state_next_inconsistent = true;
        }
    }

    /// Fix #3062: Record that this operator evaluation read TLCGet("level").
    /// If the operator reads the level multiple times and gets different values,
    /// mark as inconsistent (should not happen in practice since level is stable
    /// within a single state evaluation).
    pub(crate) fn record_tlc_level(&mut self, level: u32) {
        match self.tlc_level {
            Some(prev) if prev != level => {
                self.inconsistent = true;
            }
            Some(_) => {} // same level, no-op
            None => {
                self.tlc_level = Some(level);
            }
        }
    }

    pub(crate) fn merge_from(&mut self, other: &OpEvalDeps) {
        if other.inconsistent {
            self.inconsistent = true;
        }
        if other.state_next_inconsistent {
            self.state_next_inconsistent = true;
        }
        // Fix #3447: Propagate instance_lazy_read taint.
        if other.instance_lazy_read {
            self.instance_lazy_read = true;
        }
        for (name_id, ld) in &other.local {
            // Carry the child's record-time depth so the receiving frame's
            // `strip_internal_locals` can decide internal-vs-external against its
            // own base_stack_len using the stable global-coordinate depth.
            self.record_local_compact(*name_id, &ld.value, ld.depth);
        }
        for (idx, cv) in other.state.iter() {
            if self.state.record_compact(idx, cv) {
                self.inconsistent = true;
                self.state_next_inconsistent = true;
            }
        }
        for (idx, cv) in other.next.iter() {
            if self.next.record_compact(idx, cv) {
                self.inconsistent = true;
                self.state_next_inconsistent = true;
            }
        }
        // Fix #3062: Merge tlc_level dependency.
        if let Some(level) = other.tlc_level {
            self.record_tlc_level(level);
        }
    }

    /// Part of #2991: Strip local deps that were internal to the operator evaluation.
    ///
    /// After a dep tracking frame completes, local deps that were bound AT OR ABOVE
    /// `base_stack_len` are internal iteration variables (quantifiers, comprehensions,
    /// recursive-function indices) and must not affect cache validity.
    ///
    /// `record_local_read` correctly filters DIRECT reads using the depth check,
    /// but deps arriving via `propagate_cached_deps` (from Instance binding deps
    /// or nested operator dep tracking) bypass this check. This method applies
    /// the same filter retroactively.
    ///
    /// The decision uses each dep's `record-time` depth (`LocalDep::depth`) when
    /// known, which is the same `index < base_stack_len` basis `record_local_read`
    /// uses and is unambiguous even when an inner internal variable shadows an
    /// identically-named outer binding (the failure mode of the prior NameId
    /// re-lookup; see `LocalDep`). When the depth is unknown (`None`, reconstructed
    /// deps), it falls back to the previous binding-chain lookup.
    ///
    /// If filtering removes all local deps and state/next had no conflicts,
    /// `inconsistent` is cleared since it was caused exclusively by internal locals.
    pub(crate) fn strip_internal_locals(
        &mut self,
        bindings: &crate::binding_chain::BindingChain,
        base_stack_len: usize,
    ) {
        if self.local.is_empty() {
            return;
        }
        let base = base_stack_len as u32;
        self.local.retain(|(name_id, ld)| match ld.depth {
            // Keep the dep only if it was bound BEFORE the evaluation started
            // (depth < base) — i.e. it is an external captured variable, not an
            // internal iteration variable. Uses the stable record-time depth.
            Some(depth) => depth < base,
            // Depth unknown (reconstructed dep): fall back to the binding-chain
            // lookup, preserving the established behavior for these entries.
            None => bindings
                .lookup_local_depth(*name_id)
                .is_some_and(|depth| depth < base_stack_len),
        });
        // If all local deps were internal and state/next had no conflicts,
        // the inconsistency was caused solely by internal locals — clear it.
        if self.local.is_empty() && !self.state_next_inconsistent {
            self.inconsistent = false;
        }
    }
}
