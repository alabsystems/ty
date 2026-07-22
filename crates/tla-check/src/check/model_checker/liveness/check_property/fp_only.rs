// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::{
    check_error_to_result, Arc, CheckResult, Fingerprint, FxHashMap, LiveExpr, ModelChecker, State,
    SuccessorGraph,
};
use crate::state::ArrayState;
use crate::storage::TraceLocationStorage;
use crate::LivenessCheckError;
use rustc_hash::FxHashSet;
use std::collections::VecDeque;

/// Default maximum number of entries in the fp-only liveness replay cache.
///
/// The replay cache materializes `State` (OrdMap-based) for every reachable
/// fingerprint. Each `State` is ~200-500 bytes depending on variable count,
/// so 5M entries can consume 1-2.5 GB. Beyond this limit, the BFS replay
/// stops inserting new entries and falls back to per-state trace reconstruction
/// for the remaining states — slower but bounded in memory.
///
/// Override via the `TY_REPLAY_CACHE_MAX` environment variable.
///
/// Part of #4080: OOM safety — fp_only_replay_cache capping.
const DEFAULT_REPLAY_CACHE_MAX: usize = 5_000_000;

/// Number of missing trace-index fingerprints to include in the aggregate warning.
const MISSING_TRACE_LOCS_SAMPLE_LIMIT: usize = 5;

/// Read the max replay cache size from `TY_REPLAY_CACHE_MAX` env var,
/// falling back to [`DEFAULT_REPLAY_CACHE_MAX`].
/// Cached via `OnceLock` (Part of #4114).
fn replay_cache_max_from_env() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("TY_REPLAY_CACHE_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_REPLAY_CACHE_MAX)
    })
}

fn all_leaves_within_tag_range(
    expr: &LiveExpr,
    max_fairness_tag: u32,
    max_inline_tag: u32,
    multiword: bool,
) -> bool {
    match expr {
        LiveExpr::Bool(_) => true,
        LiveExpr::StatePred { tag, .. }
        | LiveExpr::Enabled { tag, .. }
        | LiveExpr::ActionPred { tag, .. }
        | LiveExpr::StateChanged { tag, .. } => {
            // SOUNDNESS (#4159 follow-up): the u64 first-word reconstruction
            // (`reconstruct_check_from_tag_bits`) can only represent tags 0..=63. When the bitmask
            // backend cannot serve multi-word bits, a leaf with tag >= 64 must NOT use the fast path
            // (it would be silently read as 0). Require `tag < 64` so such leaves fall back to the
            // correct eval path. When `multiword` is set, the cache reconstruction routes through
            // `storage::bitmask_map::reconstruct_check_from_bitmask` (multi-word) and the ceiling is
            // lifted.
            let within_word_limit = multiword || *tag < 64;
            *tag > 0 && within_word_limit && (*tag <= max_fairness_tag || *tag <= max_inline_tag)
        }
        LiveExpr::Not(inner) => {
            all_leaves_within_tag_range(inner, max_fairness_tag, max_inline_tag, multiword)
        }
        LiveExpr::And(exprs) | LiveExpr::Or(exprs) => exprs.iter().all(|expr| {
            all_leaves_within_tag_range(expr, max_fairness_tag, max_inline_tag, multiword)
        }),
        LiveExpr::Always(_) | LiveExpr::Eventually(_) | LiveExpr::Next(_) => false,
    }
}

pub(super) fn all_checks_structurally_cached(
    plan: &crate::liveness::GroupedLivenessPlan,
    max_fairness_tag: u32,
    max_inline_tag: u32,
    has_inline_results: bool,
    multiword: bool,
) -> bool {
    if !has_inline_results {
        return false;
    }

    let mut action_used = vec![false; plan.check_action.len()];
    let mut state_used = vec![false; plan.check_state.len()];
    for pem in &plan.pems {
        for &idx in &pem.ea_action_idx {
            if idx < action_used.len() {
                action_used[idx] = true;
            }
        }
        for &idx in &pem.ae_action_idx {
            if idx < action_used.len() {
                action_used[idx] = true;
            }
        }
        for &idx in &pem.ea_state_idx {
            if idx < state_used.len() {
                state_used[idx] = true;
            }
        }
        for &idx in &pem.ae_state_idx {
            if idx < state_used.len() {
                state_used[idx] = true;
            }
        }
    }

    let state_ok = plan.check_state.iter().enumerate().all(|(idx, check)| {
        !state_used[idx]
            || all_leaves_within_tag_range(check, max_fairness_tag, max_inline_tag, multiword)
    });
    let action_ok = plan.check_action.iter().enumerate().all(|(idx, check)| {
        !action_used[idx]
            || all_leaves_within_tag_range(check, max_fairness_tag, max_inline_tag, multiword)
    });
    state_ok && action_ok
}

impl ModelChecker<'_> {
    pub(super) fn replay_fingerprint_path(
        &mut self,
        path: &[Fingerprint],
    ) -> Result<Vec<State>, CheckResult> {
        if path.is_empty() {
            return Ok(Vec::new());
        }

        let (init_name, next_name) =
            match (&self.trace.cached_init_name, &self.trace.cached_next_name) {
                (Some(init), Some(next)) => (init.clone(), next.clone()),
                _ => {
                    return Err(check_error_to_result(
                        LivenessCheckError::RuntimeFailure(
                            "Init/Next operator names not cached for liveness counterexample replay"
                                .to_string(),
                        )
                        .into(),
                        &self.stats,
                    ));
                }
            };

        let initial_states = match self.generate_initial_states(&init_name) {
            Ok(states) => states,
            Err(error) => return Err(check_error_to_result(error, &self.stats)),
        };

        let Some(mut current_state) = initial_states.into_iter().find(|state| {
            self.state_fingerprint(state)
                .map(|fp| {
                    fp == path[0]
                        || (self.uses_compiled_bfs_fingerprint_domain()
                            && state.fingerprint() == path[0])
                })
                .unwrap_or(false)
        }) else {
            return Err(check_error_to_result(
                LivenessCheckError::RuntimeFailure(format!(
                    "could not reconstruct fingerprint path: no initial state matches {}",
                    path[0]
                ))
                .into(),
                &self.stats,
            ));
        };

        let mut states = vec![current_state.clone()];
        for &target_fp in &path[1..] {
            let successors = match self.solve_next_relation(&next_name, &current_state) {
                Ok(successors) => successors,
                Err(error) => return Err(check_error_to_result(error, &self.stats)),
            };

            let mut matched_state = None;
            for successor in successors {
                let successor_fp = match self.state_fingerprint(&successor) {
                    Ok(fp) => fp,
                    Err(error) => return Err(check_error_to_result(error, &self.stats)),
                };
                if successor_fp == target_fp {
                    matched_state = Some(successor);
                    break;
                }
            }

            let Some(next_state) = matched_state else {
                return Err(check_error_to_result(
                    LivenessCheckError::RuntimeFailure(format!(
                        "counterexample replay could not find transition {} -> {}",
                        current_state.fingerprint(),
                        target_fp
                    ))
                    .into(),
                    &self.stats,
                ));
            };

            states.push(next_state.clone());
            current_state = next_state;
        }

        Ok(states)
    }

    /// Opportunistically capture one completed BFS state for the fp-only
    /// liveness state cache (#liveness-bfs-state-seed).
    ///
    /// Called from `record_inline_liveness_results` with the dequeued state's
    /// BFS fingerprint and array — exactly the `(fp, state)` pair the post-BFS
    /// replay in `build_fp_only_liveness_state_cache` would otherwise
    /// reconstruct by re-enumerating the Next relation over the whole
    /// reachable graph (profiled at ~11% of total CPU on Huang).
    ///
    /// Exactness: the entry stores the very array whose fingerprint the BFS
    /// computed, under that fingerprint — the same fp domain the successor
    /// graph and the inline bitmask keys use — so a seeded entry is
    /// definitionally the state the replay's `array_state_fingerprint`
    /// matching would have selected. Missing entries (cap reached, gate
    /// inactive, early BFS stop) fall back to the existing replay/per-state
    /// reconstruction paths untouched (fail-closed).
    ///
    /// Memory: gated to runs where a tableau-carrying inline property plan
    /// guarantees `needs_full_state_cache` (the `!no_tableau_fast_path`
    /// branch is unconditional), i.e. the fp-only phase would build this very
    /// map anyway; seeding only moves that allocation earlier. VIEW and
    /// SYMMETRY runs are excluded (their cache build has extra canonical-fp
    /// handling this fast path does not reproduce).
    pub(in crate::check::model_checker) fn maybe_seed_fp_only_state_cache(
        &mut self,
        fp: Fingerprint,
        array: &ArrayState,
    ) {
        if self.liveness_cache.fp_only_replay_cache.is_some() {
            return;
        }
        if !matches!(
            self.liveness_mode,
            super::super::LivenessMode::FingerprintOnly { view: false }
        ) {
            return;
        }
        if !self.symmetry.perms.is_empty() {
            return;
        }
        // Post-BFS liveness must actually be coming (successor graph is being
        // captured for it). Runs whose properties end up served entirely by
        // the inline bitmasks never drain the seed; it is freed right after
        // the liveness phase (`discard_unused_fp_only_seed`), so the
        // over-approximation costs bounded transient memory — the same map,
        // at the same `TY_REPLAY_CACHE_MAX` cap, that the replay path would
        // materialize for every spec that does need it.
        if !self.liveness_cache.cache_for_liveness {
            return;
        }
        if self.liveness_cache.bfs_seeded_states.len() >= replay_cache_max_from_env() {
            return;
        }
        self.liveness_cache
            .bfs_seeded_states
            .entry(fp)
            .or_insert_with(|| array.clone());
    }

    /// Drop any BFS-time seed the liveness phase did not consume
    /// (#liveness-bfs-state-seed). Called once after liveness checking so
    /// bitmask-fast-path runs do not retain the speculative state map.
    pub(in crate::check::model_checker) fn discard_unused_fp_only_seed(&mut self) {
        if !self.liveness_cache.bfs_seeded_states.is_empty() {
            self.liveness_cache.bfs_seeded_states = Default::default();
        }
    }

    /// Build the full state cache for fp-only liveness checking using BFS-order
    /// replay. Cached across properties to avoid redundant replay per property.
    ///
    /// Part of #3210: The old implementation called `reconstruct_trace(fp)` for each
    /// of S states, replaying the full Init→...→state path per state — O(S×D) total
    /// work (D = avg depth), called N times (once per property). The new implementation
    /// does a single BFS from init states through `cached_successors`, reconstructing
    /// each state exactly once via Next-relation evaluation — O(S) total, called once.
    /// Matches the parallel checker's `replay_fp_only_state_cache()` pattern.
    #[allow(clippy::type_complexity)]
    pub(super) fn build_fp_only_liveness_state_cache(
        &mut self,
        init_fps: &[Fingerprint],
        cached_successors: &SuccessorGraph,
    ) -> Result<
        (
            Arc<FxHashMap<Fingerprint, ArrayState>>,
            Arc<FxHashMap<Fingerprint, Fingerprint>>,
        ),
        CheckResult,
    > {
        // Return cached result if already computed for a previous property.
        if let Some(ref cached) = self.liveness_cache.fp_only_replay_cache {
            return Ok(cached.clone());
        }

        let registry = self.ctx.var_registry().clone();
        let cache_max = replay_cache_max_from_env();

        // Step 1: Seed from init states in the liveness cache (small, typically 1-10).
        // Cap pre-allocation to avoid over-reserving when successor graph is huge.
        // Stores compact ArrayStates (not im::OrdMap State) — the behavior graph
        // reconstructs State lazily only for trace output.
        let prealloc = (init_fps.len() + cached_successors.len()).min(cache_max);
        let mut state_cache: FxHashMap<Fingerprint, ArrayState> =
            FxHashMap::with_capacity_and_hasher(prealloc, Default::default());
        let mut queue: VecDeque<Fingerprint> = VecDeque::new();

        for (fp, arr) in &self.liveness_cache.init_states {
            state_cache.insert(*fp, arr.clone());
            queue.push_back(*fp);
        }

        // Step 1.5 (#liveness-bfs-state-seed): adopt the states captured
        // during BFS. Every seeded fp is also enqueued so the replay walk can
        // traverse THROUGH pre-cached states into any unseeded region (cap
        // overflow / early stop) — a seeded state's successors are checked
        // exactly like a freshly reconstructed one's. When the seed is
        // complete, every pop finds `needed` empty and the walk performs no
        // Next-relation generation at all.
        let seeded_count = self.liveness_cache.bfs_seeded_states.len();
        if seeded_count > 0 {
            let seeded = std::mem::take(&mut self.liveness_cache.bfs_seeded_states);
            for (fp, arr) in seeded {
                if state_cache.len() >= cache_max {
                    break;
                }
                if let std::collections::hash_map::Entry::Vacant(v) = state_cache.entry(fp) {
                    v.insert(arr);
                    queue.push_back(fp);
                }
            }
        }
        if crate::liveness::debug::liveness_profile() {
            eprintln!(
                "[fp-only] state cache build: {} BFS-seeded states adopted",
                seeded_count
            );
        }

        // Step 2: BFS from init states through cached_successors, replaying
        // Next-relation via ArrayState-based generation (DiffSuccessor streaming
        // path) instead of the slow State-based `solve_next_relation`. This avoids
        // O(n) OrdMap construction per successor — only matched successors are
        // converted to State via `to_state()`. Part of #3739.
        //
        // Part of #4080: Stop BFS when cache exceeds `cache_max` entries. The
        // remaining unreached states will be handled by the per-state fallback
        // in Step 3, which is slower but bounded in memory (one state at a time).

        while let Some(parent_fp) = queue.pop_front() {
            // Part of #4080: Stop BFS replay when cache is at capacity.
            if state_cache.len() >= cache_max {
                eprintln!(
                    "Warning: fp-only liveness replay cache capped at {cache_max} entries. \
                     Remaining states will use per-state trace reconstruction \
                     (slower but memory-bounded). Set TY_REPLAY_CACHE_MAX to \
                     adjust this limit."
                );
                break;
            }

            // Part of #4080: Use get_ref() to avoid cloning the entire Vec<Fingerprint>
            // on every lookup in the in-memory backend. Falls back to get() for disk.
            let owned_fallback;
            let expected_succs: &[Fingerprint] =
                if let Some(s) = cached_successors.get_ref(&parent_fp) {
                    s
                } else {
                    owned_fallback = cached_successors.get(&parent_fp);
                    match owned_fallback.as_deref() {
                        Some(s) => s,
                        None => continue,
                    }
                };
            if expected_succs.is_empty() {
                continue;
            }

            // Collect only successors not yet in cache.
            let needed: Vec<Fingerprint> = expected_succs
                .iter()
                .filter(|fp| !state_cache.contains_key(fp))
                .copied()
                .collect();
            if needed.is_empty() {
                continue;
            }

            // Cache already stores ArrayState — use directly (no State round-trip).
            let parent_array = match state_cache.get(&parent_fp) {
                Some(s) => s.clone(),
                None => continue,
            };

            // Use ArrayState-based successor generation which avoids State/OrdMap
            // construction overhead. Falls back to State-based path on error.
            let succ_arrays = self
                .generate_successors_as_array(&parent_array)
                .map_err(|e| check_error_to_result(e, &self.stats))?;

            let mut needed_set: FxHashSet<Fingerprint> =
                FxHashSet::with_capacity_and_hasher(needed.len(), Default::default());
            needed_set.extend(needed.iter().copied());

            for mut succ_array in succ_arrays {
                let succ_fp = self
                    .array_state_fingerprint(&mut succ_array)
                    .map_err(|e| check_error_to_result(e, &self.stats))?;
                if needed_set.remove(&succ_fp) {
                    // Store the compact ArrayState for matched successors; State
                    // is reconstructed lazily only for trace output.
                    state_cache.insert(succ_fp, succ_array);
                    queue.push_back(succ_fp);
                }
                if needed_set.is_empty() {
                    break;
                }
                // Part of #4080: Check cache cap within inner loop too.
                if state_cache.len() >= cache_max {
                    break;
                }
            }
        }

        // Step 3: Fallback for any states not reached by BFS replay.
        // This can happen when fingerprinting produces different results during
        // replay vs BFS (e.g., evaluation caching, interner state). Fall back to
        // per-state trace reconstruction for just the missing states.
        let mut all_expected = cached_successors.collect_all_fingerprints();
        all_expected.extend(init_fps.iter().copied());
        let missing: Vec<Fingerprint> = all_expected
            .into_iter()
            .filter(|fp| !state_cache.contains_key(fp))
            .collect();
        if !missing.is_empty() {
            self.trace.ensure_trace_index_built();
            let mut missing_trace_locs = 0usize;
            let mut missing_trace_locs_sample = Vec::new();

            for fp in &missing {
                if !self.trace.trace_locs.contains(fp) {
                    missing_trace_locs += 1;
                    if missing_trace_locs_sample.len() < MISSING_TRACE_LOCS_SAMPLE_LIMIT {
                        missing_trace_locs_sample.push(*fp);
                    }
                    continue;
                }

                let trace = self.reconstruct_trace(*fp);
                if let Some(state) = trace.states.last() {
                    state_cache.insert(*fp, ArrayState::from_state(state, &registry));
                }
            }

            if missing_trace_locs > 0 {
                eprintln!(
                    "WARNING: skipped fp-only liveness trace reconstruction for {} \
                     fingerprint(s) absent from the trace location index; sample: {:?}",
                    missing_trace_locs, missing_trace_locs_sample
                );
            }
        }

        let mut state_fp_to_canon_fp: FxHashMap<Fingerprint, Fingerprint> =
            FxHashMap::with_capacity_and_hasher(state_cache.len(), Default::default());
        for (canon_fp, arr) in &state_cache {
            // Transient to_state (dropped) to reproduce the exact
            // State::fingerprint() the canon map historically keyed on.
            state_fp_to_canon_fp.insert(arr.to_state(&registry).fingerprint(), *canon_fp);
        }

        let result = (Arc::new(state_cache), Arc::new(state_fp_to_canon_fp));
        self.liveness_cache.fp_only_replay_cache = Some(result.clone());
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check::{resolve_spec_from_config, CheckResult};
    use crate::Config;
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    const FP_ONLY_FALLBACK_SPEC: &str = r#"
---- MODULE FpOnlyFallbackReplay ----
EXTENDS Integers

VARIABLE x

Init == x = 0

Next ==
    \/ /\ x = 0
       /\ x' = 1
    \/ /\ x = 1
       /\ x' = 2
    \/ /\ x = 2
       /\ x' = 2

Spec == Init /\ [][Next]_x /\ WF_x(Next)

EventuallyTwo == <> (x = 2)
====
"#;

    fn run_fp_only_checker(use_disk_successors: bool, extra_missing_trace_fps: &[Fingerprint]) {
        let tree = parse_to_syntax_tree(FP_ONLY_FALLBACK_SPEC);
        let module = lower(FileId(0), &tree).module.expect("lowered module");
        let unresolved = Config {
            specification: Some("Spec".to_string()),
            ..Default::default()
        };
        let resolved =
            resolve_spec_from_config(&unresolved, &tree).expect("SPECIFICATION should resolve");
        let config = Config {
            init: Some(resolved.init.clone()),
            next: Some(resolved.next.clone()),
            specification: unresolved.specification.clone(),
            properties: vec!["EventuallyTwo".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        if use_disk_successors {
            checker.liveness_cache.successors =
                SuccessorGraph::disk().expect("disk successor graph should initialize");
        }
        checker.set_deadlock_check(false);
        checker.set_store_states(false);
        checker.set_fairness(resolved.fairness);
        checker.set_stuttering_allowed(resolved.stuttering_allowed);

        match checker.check() {
            CheckResult::Success(stats) => {
                assert_eq!(stats.states_found, 3, "expected 3 reachable states");
            }
            other => panic!("expected liveness success, got: {other:?}"),
        }

        let init_fps: Vec<Fingerprint> = checker
            .liveness_cache
            .init_states
            .iter()
            .map(|(fp, _)| *fp)
            .collect();
        assert_eq!(init_fps.len(), 1, "expected one initial state");

        checker.liveness_cache.fp_only_replay_cache = None;
        checker.liveness_cache.init_states.clear();

        let mut cached_successors = std::mem::take(&mut checker.liveness_cache.successors);
        for fp in extra_missing_trace_fps {
            cached_successors
                .insert(*fp, Vec::new())
                .expect("test should be able to inject missing trace-index fingerprints");
        }
        let rebuilt = checker
            .build_fp_only_liveness_state_cache(&init_fps, &cached_successors)
            .expect("fallback replay should rebuild the full state cache");
        checker.liveness_cache.successors = cached_successors;

        assert_eq!(
            rebuilt.0.len(),
            3,
            "fallback replay should rebuild all three states when the init-state seed is absent"
        );
        for fp in extra_missing_trace_fps {
            assert!(
                !rebuilt.0.contains_key(fp),
                "fallback replay should skip fingerprints absent from the trace location index"
            );
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn fp_only_replay_fallback_rebuilds_state_cache_with_in_memory_successors() {
        run_fp_only_checker(false, &[]);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn fp_only_replay_fallback_rebuilds_state_cache_with_disk_successors() {
        run_fp_only_checker(true, &[]);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn fp_only_replay_fallback_skips_fingerprints_absent_from_trace_index() {
        run_fp_only_checker(false, &[Fingerprint(0xffff_ffff_ffff_ff00)]);
    }
}
