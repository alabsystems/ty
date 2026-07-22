// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Full-state BFS storage backend.
//!
//! Part of #3436: extracted from `storage_modes.rs`.

use super::super::super::frontier::BfsFrontier;
use super::super::super::{ArrayState, CheckResult, Fingerprint, ModelChecker, State, VecDeque};
use super::super::checkpoint_view;
use super::BfsStorage;

/// Full-state BFS storage: states live in a `FxHashMap<Fingerprint, ArrayState>`.
///
/// Queue entries are fingerprints — the actual state is retrieved from the HashMap.
/// This mode supports trace reconstruction via parent pointers.
pub(in crate::check::model_checker) struct FullStateStorage;

impl BfsStorage for FullStateStorage {
    /// Queue entries carry `(Fingerprint, depth)` — the depth travels with the
    /// entry rather than being looked up from `trace.depths` at dequeue time.
    ///
    /// Part of #2881 Step 3: eliminates the per-dequeue `FxHashMap::get` on the
    /// `depths` HashMap, matching TLC's approach of carrying per-state metadata
    /// on the state/queue object rather than in a side table.
    type QueueEntry = (Fingerprint, usize);

    fn dequeue(
        &mut self,
        (fp, depth): (Fingerprint, usize),
        mc: &mut ModelChecker,
    ) -> Result<Option<(Fingerprint, ArrayState, usize)>, CheckResult> {
        let current_array = match mc.state_storage.seen.remove(&fp) {
            Some(arr) => arr,
            None => {
                mc.stats.phantom_dequeues += 1;
                return Ok(None);
            }
        };
        Ok(Some((fp, current_array, depth)))
    }

    fn return_current(&mut self, fp: Fingerprint, state: ArrayState, mc: &mut ModelChecker) {
        mc.state_storage.seen.insert(fp, state);
    }

    fn admit_successor(
        &mut self,
        fp: Fingerprint,
        state: ArrayState,
        parent_fp: Option<Fingerprint>,
        current: Option<(Fingerprint, &ArrayState)>,
        depth: usize,
        mc: &mut ModelChecker,
    ) -> Result<Option<(Fingerprint, usize)>, CheckResult> {
        if mc.mark_state_seen_owned_checked_with_current(fp, state, parent_fp, depth, current)? {
            Ok(Some((fp, depth)))
        } else {
            Ok(None)
        }
    }

    fn enforce_seen_successor_duplicate(
        &mut self,
        fp: Fingerprint,
        candidate: &ArrayState,
        current: Option<(Fingerprint, &ArrayState)>,
        mc: &mut ModelChecker,
    ) -> Result<(), CheckResult> {
        mc.enforce_seen_state_duplicate_with_payload(fp, candidate, current)
    }

    fn use_diffs(&self, mc: &ModelChecker) -> bool {
        if crate::check::debug::force_no_diffs() {
            return false;
        }
        // Nested-set A6: the diff/streaming fast-path is NO LONGER disabled when a
        // monitor is installed. The per-successor escape monitor is now hooked
        // INTO the streaming ClosureSink (`observe_diff_monitors_escape`), so it
        // still sees EVERY successor's board and fails closed on escape — without
        // forcing the slow batch path. Verdict-identical (the monitor never
        // changes a fingerprint; the diff path already produces
        // `value_fingerprint(board)`, which byte-matches the monitored `dedup_fp`).
        mc.compiled.cached_view_name.is_none()
            && mc.symmetry.perms.is_empty()
            && !mc.inline_liveness_active()
    }

    fn checkpoint_frontier(
        &self,
        current: &ArrayState,
        queue: &impl BfsFrontier<Entry = (Fingerprint, usize)>,
        registry: &crate::var_index::VarRegistry,
        mc: &mut ModelChecker,
    ) -> VecDeque<State> {
        checkpoint_view::build_checkpoint_frontier(current, queue, registry, |(q_fp, _depth)| {
            mc.state_storage
                .seen
                .get(q_fp)
                .map(|arr| arr.to_state(registry))
                .or_else(|| {
                    // Queue fingerprints must survive checkpointing even when the
                    // full-state map no longer retains their ArrayState. Rebuild
                    // the state from the trace path instead of silently dropping
                    // the frontier entry.
                    mc.reconstruct_trace(*q_fp).states.last().cloned()
                })
        })
    }

    fn cache_diff_liveness(
        &self,
        parent_fp: Fingerprint,
        succ_fps: Option<Vec<Fingerprint>>,
        mc: &mut ModelChecker,
    ) -> Result<(), crate::check::CheckError> {
        // Symmetry is excluded from the diff path, so we only cache fingerprints.
        if let Some(fps) = succ_fps {
            mc.liveness_cache.successors.insert(parent_fp, fps)?;
        }
        Ok(())
    }

    fn cache_full_liveness(
        &self,
        parent_fp: Fingerprint,
        successors: &[(ArrayState, Fingerprint)],
        mc: &mut ModelChecker,
    ) -> Result<(), crate::check::CheckError> {
        if !mc.liveness_cache.cache_for_liveness {
            return Ok(());
        }
        let succ_fps: Vec<Fingerprint> = successors.iter().map(|(_, fp)| *fp).collect();
        mc.liveness_cache.successors.insert(parent_fp, succ_fps)?;
        if !mc.symmetry.perms.is_empty() {
            // 2026-07 memory audit: intern witness states by their raw content
            // fingerprint so duplicate concrete successors (the overwhelmingly
            // common case: BufferedRandomAccessFile generates 248,697
            // transitions over 6,376 distinct states) share ONE retained
            // allocation instead of one per transition. Equal raw fps are
            // trusted as equal values — the same fp64 trust the BFS dedup and
            // SUBSCRIPT_VALUE_CACHE already place in content fingerprints; a
            // collision substitutes a value-equal-by-fp witness candidate and
            // every downstream action check still evaluates the actual values.
            // Kill switch: TY_NO_WITNESS_INTERN=1 (restores per-transition
            // clones).
            let intern = !crate::check::debug::no_witness_intern();
            let mut witness_list: Vec<(Fingerprint, ArrayState)> =
                Vec::with_capacity(successors.len());
            for (arr, canon_fp) in successors {
                let shared = if intern {
                    let raw_fp =
                        super::super::super::liveness::compute_fingerprint_from_compact_values(
                            arr.values(),
                            mc.ctx.var_registry(),
                        );
                    mc.liveness_cache
                        .witness_intern
                        .entry(raw_fp)
                        .or_insert_with(|| arr.clone())
                        .clone()
                } else {
                    arr.clone()
                };
                witness_list.push((*canon_fp, shared));
            }
            mc.liveness_cache
                .successor_witnesses
                .insert(parent_fp, witness_list);
        }
        Ok(())
    }
}
