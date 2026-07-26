// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Post-BFS liveness runner.
//!
//! Owns the preparation (cache building, symmetry maps), property iteration
//! loop, and the `run_liveness_checking` orchestrator. Extracted from
//! `liveness.rs` per #3605.

#[cfg(debug_assertions)]
use super::super::debug::debug_safety_temporal;
use super::super::debug::{liveness_profile, skip_liveness};
use super::super::{
    Arc, ArrayState, CheckResult, Fingerprint, FxHashMap, ModelChecker,
    SafetyTemporalPropertyOutcome, SuccessorWitnessMap,
};
use super::{check_property, compute_fingerprint_from_compact_values, LivenessMode};
use crate::storage::{
    ActionBitmaskMap, StateBitmaskMap, SuccessorGraph, TraceLocationStorage, TraceLocationsStorage,
};

#[allow(clippy::fn_params_excessive_bools)]
pub(super) const fn exact_otf_owned_cache_admitted(
    partial_graph: bool,
    on_the_fly: bool,
    has_view: bool,
    has_symmetry: bool,
    compact_cache_enabled: bool,
    init_state_count: usize,
    expected_initial_states: usize,
    states_found: usize,
) -> bool {
    !partial_graph
        && on_the_fly
        && !has_view
        && !has_symmetry
        && compact_cache_enabled
        && init_state_count == expected_initial_states
        && init_state_count <= states_found
        && (states_found == 0 || init_state_count != 0)
}

impl ModelChecker<'_> {
    /// Build the raw-state-fingerprint -> canonical graph fingerprint map.
    ///
    /// Part of #2661: compute raw fingerprints directly from ArrayState values
    /// so symmetry setup no longer needs eager ArrayState -> State conversion.
    pub(super) fn build_state_fp_to_canon_fp(
        &self,
        registry: &crate::var_index::VarRegistry,
    ) -> FxHashMap<Fingerprint, Fingerprint> {
        let mut fp_map: FxHashMap<Fingerprint, Fingerprint> =
            FxHashMap::with_capacity_and_hasher(self.state_storage.seen.len(), Default::default());
        for (canon_fp, arr) in &self.state_storage.seen {
            let raw_fp = compute_fingerprint_from_compact_values(arr.values(), registry);
            fp_map.insert(raw_fp, *canon_fp);
        }
        fp_map
    }

    /// Convert cached BFS witness arrays into a post-BFS liveness witness map.
    ///
    /// Under SYMMETRY, action-level liveness checks must run against the actual
    /// successor states produced during BFS rather than the reduced
    /// representative recovered later from `seen`. This mirrors TLC's
    /// `LiveCheck.addNextState` contract and the parallel checker path.
    ///
    /// Part of #2661: Stores ArrayState directly instead of converting to State,
    /// eliminating the O(W) `to_state` conversion. Consumers use `values()` for
    /// O(1) array-backed binding. Raw fingerprints are recomputed from the array
    /// values during the build loop for `state_fp_to_canon_fp` registration.
    pub(super) fn build_successor_witnesses(
        &self,
        registry: &crate::var_index::VarRegistry,
        state_fp_to_canon_fp: &mut FxHashMap<Fingerprint, Fingerprint>,
    ) -> Option<Arc<SuccessorWitnessMap>> {
        if self.liveness_cache.successor_witnesses.is_empty() {
            return None;
        }

        let mut out: SuccessorWitnessMap = FxHashMap::with_capacity_and_hasher(
            self.liveness_cache.successor_witnesses.len(),
            Default::default(),
        );
        for (from_fp, witness_list) in &self.liveness_cache.successor_witnesses {
            let mut converted = Vec::with_capacity(witness_list.len());
            for (dest_canon_fp, arr) in witness_list {
                // Part of #2661: Compute the raw fingerprint from ArrayState
                // values directly. This avoids populating fp_cache on a cloned
                // witness just to register the raw->canonical map.
                let raw_fp = compute_fingerprint_from_compact_values(arr.values(), registry);
                state_fp_to_canon_fp.entry(raw_fp).or_insert(*dest_canon_fp);
                converted.push((*dest_canon_fp, arr.clone()));
            }
            out.insert(*from_fp, converted);
        }

        Some(Arc::new(out))
    }

    fn should_skip_promoted_liveness_property(&self, prop_name: &str) -> bool {
        self.is_property_fully_promoted(prop_name)
    }

    /// Release BFS payload indexes once complete retained Init roots and the
    /// snapshotted exact on-the-fly route guarantee that this liveness pass can
    /// regenerate and own every concrete payload needed for verdicts and
    /// counterexamples.
    ///
    /// Keep the fingerprint backend object and trace file alive: terminal
    /// storage-error precedence still consults the former, while the latter is
    /// a cheap fallback artifact if a future cold path needs it. A uniquely
    /// owned, opt-in backend may release only its membership entries after the
    /// authoritative count and storage statistics have been snapshotted.
    fn release_exact_otf_bfs_payloads(&mut self) {
        let seen = std::mem::take(&mut self.state_storage.seen);
        let trace_locs = std::mem::replace(
            &mut self.trace.trace_locs,
            TraceLocationsStorage::in_memory(),
        );
        // Full-state BFS normally maintains this index eagerly. After retiring
        // it, allow an unforeseen cold trace consumer to reconstruct it from
        // the retained trace file instead of treating the empty index as final.
        self.trace.lazy_trace_index = true;
        let seen_len = seen.len();
        let trace_loc_len = trace_locs.len();
        drop(seen);
        drop(trace_locs);
        let fingerprint_len = self
            .state_storage
            .try_release_terminal_seen_fps_entries(self.stats.states_found);

        if liveness_profile() {
            eprintln!(
                "[liveness] released exact-OTF BFS payloads: {seen_len} state witnesses, \
                 {trace_loc_len} trace locations, {fingerprint_len} fingerprint membership \
                 entries (fingerprint backend retained)"
            );
        }
    }

    fn run_liveness_properties_on_the_fly(&mut self, partial_graph: bool) -> Option<CheckResult> {
        if partial_graph {
            return None;
        }

        // Detach the potentially wide compact Init vector and move the retained
        // graph into a run-scoped session. Mutable successor closures can then
        // replay a complete retained source or evaluate Next without aliasing
        // `self`, and the session can release redundant storage between groups.
        let init_states = std::mem::take(&mut self.liveness_cache.init_states);
        // Snapshot this routing decision once. In particular, do not re-read
        // the compact-cache kill switch after dropping the only BFS payload
        // indexes: exact exploration must remain on the owned-cache route for
        // this entire liveness pass.
        let use_owned_compact_cache = exact_otf_owned_cache_admitted(
            partial_graph,
            self.should_run_on_the_fly_liveness(),
            self.compiled.cached_view_name.is_some(),
            !self.symmetry.perms.is_empty(),
            crate::liveness::debug::liveness_otf_compact_cache_enabled(),
            init_states.len(),
            self.stats.initial_states,
            self.stats.states_found,
        );
        if use_owned_compact_cache {
            self.release_exact_otf_bfs_payloads();
        }
        let retained_successors = std::mem::take(&mut self.liveness_cache.successors);
        let (result, retained_successors) =
            if self.liveness_cache.regenerate_on_the_fly && retained_successors.len() != 0 {
                let mut retained =
                    check_property::OtfRetainedSuccessors::new(retained_successors, &init_states);
                let result = self.run_liveness_properties_on_the_fly_with_init_states(
                    &init_states,
                    Some(&mut retained),
                    use_owned_compact_cache,
                );
                (result, retained.into_graph().unwrap_or_default())
            } else {
                let result = self.run_liveness_properties_on_the_fly_with_init_states(
                    &init_states,
                    None,
                    use_owned_compact_cache,
                );
                (result, retained_successors)
            };
        debug_assert!(self.liveness_cache.init_states.is_empty());
        debug_assert_eq!(self.liveness_cache.successors.len(), 0);
        self.liveness_cache.init_states = init_states;
        self.liveness_cache.successors = retained_successors;
        result
    }

    fn run_liveness_properties_on_the_fly_with_init_states(
        &mut self,
        init_states: &[(Fingerprint, ArrayState)],
        mut retained_successors: Option<&mut check_property::OtfRetainedSuccessors<'_>>,
        use_owned_compact_cache: bool,
    ) -> Option<CheckResult> {
        // The exact-raw cross-group cache is run-scoped and drops here. The
        // caller restores retained BFS adjacency unless an admitted exact-raw
        // cache covers it completely and lets this invocation release it early.
        let mut exact_raw_cache = check_property::OtfExactRawCacheSession::default();
        for prop_name in self.config.properties.clone() {
            if self.should_skip_promoted_liveness_property(&prop_name) {
                continue;
            }
            if let Some(result) = self.check_liveness_property_on_the_fly(
                &prop_name,
                init_states,
                retained_successors.as_deref_mut(),
                &mut exact_raw_cache,
                use_owned_compact_cache,
            ) {
                return Some(result);
            }
        }

        None
    }

    fn ensure_liveness_state_cache(
        &self,
        registry: &crate::var_index::VarRegistry,
        state_cache: &mut Arc<FxHashMap<Fingerprint, crate::state::ArrayState>>,
        state_cache_deferred: &mut bool,
        state_cache_time: &mut std::time::Duration,
    ) {
        if !*state_cache_deferred {
            return;
        }

        let sc_start = std::time::Instant::now();
        // Store compact ArrayStates; the behavior graph reconstructs `State`
        // lazily only for trace output. Cheap Arc-bump clone (no im::OrdMap).
        let _ = registry;
        let mut raw: FxHashMap<Fingerprint, crate::state::ArrayState> =
            FxHashMap::with_capacity_and_hasher(self.state_storage.seen.len(), Default::default());
        for (fp, arr) in &self.state_storage.seen {
            raw.insert(*fp, arr.clone());
        }
        *state_cache = Arc::new(raw);
        *state_cache_deferred = false;
        *state_cache_time = sc_start.elapsed();
        if liveness_profile() {
            eprintln!(
                "  state_cache build:   {:.3}s ({} states -> State, deferred/lazy)",
                state_cache_time.as_secs_f64(),
                state_cache.len()
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_liveness_properties(
        &mut self,
        partial_graph: bool,
        registry: &crate::var_index::VarRegistry,
        init_state_fps: &[Fingerprint],
        cached_successors: &SuccessorGraph,
        succ_witnesses: Option<&Arc<SuccessorWitnessMap>>,
        state_cache: &mut Arc<FxHashMap<Fingerprint, crate::state::ArrayState>>,
        state_fp_to_canon_fp: &Arc<FxHashMap<Fingerprint, Fingerprint>>,
        state_cache_deferred: &mut bool,
        state_cache_time: &mut std::time::Duration,
        cross_state_bitmasks: &StateBitmaskMap,
        cross_action_bitmasks: &ActionBitmaskMap,
    ) -> Option<CheckResult> {
        debug_eprintln!(
            debug_safety_temporal(),
            "[DEBUG] About to check {} properties",
            self.config.properties.len()
        );
        crate::eval::clear_for_phase_boundary();
        if liveness_profile() {
            eprintln!(
                "[liveness] checking {} properties...",
                self.config.properties.len()
            );
        }

        for prop_name in &self.config.properties {
            if self.should_skip_promoted_liveness_property(prop_name) {
                continue;
            }

            debug_eprintln!(
                debug_safety_temporal(),
                "[DEBUG] Checking property: {}",
                prop_name
            );
            let safety_temp_start = std::time::Instant::now();
            match self.check_safety_temporal_property(
                prop_name,
                init_state_fps,
                cached_successors,
                succ_witnesses,
            ) {
                SafetyTemporalPropertyOutcome::NotApplicable => {
                    if partial_graph {
                        if liveness_profile() {
                            eprintln!(
                                "  [periodic-liveness] Skipping non-safety-temporal property {prop_name} (partial graph)"
                            );
                        }
                        continue;
                    }
                }
                SafetyTemporalPropertyOutcome::Satisfied => {
                    if liveness_profile() {
                        eprintln!(
                            "  check_safety_temporal_property({}): {:.3}s (satisfied)",
                            prop_name,
                            safety_temp_start.elapsed().as_secs_f64()
                        );
                    }
                    continue;
                }
                SafetyTemporalPropertyOutcome::Violated(result) => {
                    return Some(*result);
                }
            }
            if liveness_profile() {
                eprintln!(
                    "  check_safety_temporal_property({}): {:.3}s",
                    prop_name,
                    safety_temp_start.elapsed().as_secs_f64()
                );
            }

            self.ensure_liveness_state_cache(
                registry,
                state_cache,
                state_cache_deferred,
                state_cache_time,
            );
            let ctx = check_property::LivenessPropertyCtx {
                init_fps: init_state_fps,
                cached_successors,
                state_cache,
                state_fp_to_canon_fp,
                succ_witnesses,
                cross_state_bitmasks,
                cross_action_bitmasks,
            };
            if let Some(result) = self.check_liveness_property(prop_name, &ctx) {
                return Some(result);
            }
        }

        None
    }

    /// Run liveness checking after BFS exploration completes.
    ///
    /// Builds successor maps, then checks each temporal property via
    /// safety-temporal decomposition and SCC-based liveness.
    ///
    /// Full `State` materialization is deferred to the SCC/tableau paths that
    /// still require it.
    ///
    /// When `partial_graph` is true (mid-BFS periodic checks), properties where
    /// `check_safety_temporal_property` returns `NotApplicable` are skipped
    /// rather than falling through to `check_liveness_property`. This prevents
    /// false positives on incomplete state graphs for properties that cannot be
    /// decomposed into safety-temporal form (e.g., `LSpec == L!Spec` resolved
    /// via ModuleRef). The final post-BFS call with `partial_graph = false`
    /// performs full SCC-based liveness checking on the complete graph.
    ///
    /// Returns `Some(result)` on violation, `None` if all properties satisfied.
    pub(in crate::check::model_checker) fn run_liveness_checking(
        &mut self,
        partial_graph: bool,
    ) -> Option<CheckResult> {
        let has_liveness_work = self.liveness_mode.is_active();
        let (fingerprint_only_mode, has_symmetry) = match self.liveness_mode {
            LivenessMode::Disabled => (false, false),
            LivenessMode::FullState { symmetry, .. } => (false, symmetry),
            LivenessMode::FingerprintOnly { .. } => (true, false),
        };
        let skip_liveness = skip_liveness();

        debug_eprintln!(
            debug_safety_temporal(),
            "[DEBUG] skip_liveness={}, fingerprint_only_mode={}, has_liveness_work={}, properties.len()={}",
            skip_liveness,
            fingerprint_only_mode,
            has_liveness_work,
            self.config.properties.len()
        );

        if !has_liveness_work {
            return None;
        }

        if skip_liveness {
            return None;
        }

        if self.should_run_on_the_fly_liveness() {
            return self.run_liveness_properties_on_the_fly(partial_graph);
        }

        // Part of #3175: Liveness checking now works in fingerprint-only mode
        // when BFS-time liveness data was collected (cache_for_liveness = true).
        if !self.liveness_cache.cache_for_liveness {
            if fingerprint_only_mode {
                // No BFS liveness data collected — emit warning for remaining temporal properties.
                let mut promoted: Vec<String> = self.compiled.promoted_property_invariants.clone();
                promoted.extend(
                    self.compiled
                        .promoted_implied_action_properties
                        .iter()
                        .cloned(),
                );
                if let Some(warning) = self.config.fingerprint_liveness_warning(false, &promoted) {
                    eprintln!("{warning}");
                }
            }
            return None;
        }

        let liveness_prep_start = std::time::Instant::now();
        let registry = self.ctx.var_registry().clone();

        // Part of #2661: Defer O(N) state_cache construction until the SCC path
        // actually needs full State objects. check_safety_temporal_property reads
        // ArrayState directly from self.state_storage.seen, and the symmetry
        // raw->canonical fingerprint map can also be built from ArrayState values.
        //
        // When deferred, the O(N) ArrayState→State conversion is skipped entirely
        // if all properties are satisfied by the safety-temporal fast path.
        let mut state_cache: Arc<FxHashMap<Fingerprint, crate::state::ArrayState>> =
            Arc::new(FxHashMap::default());
        let mut state_cache_time = std::time::Duration::ZERO;
        let mut symmetry_fp_map_time = std::time::Duration::ZERO;

        // Part of #3225: derive state_cache_deferred from LivenessMode instead
        // of reading the raw store_full_states boolean. FullState modes can
        // defer O(N) ArrayState→State conversion; FingerprintOnly never has
        // full states to build from.
        let mut state_cache_deferred = self.liveness_mode.stores_full_states();

        let (state_fp_to_canon_fp, succ_witnesses) = if has_symmetry {
            // Part of #2661: Even under symmetry, build the raw->canonical map
            // from ArrayState directly and defer the full State cache until the
            // SCC path actually needs it.
            let fp_map_start = std::time::Instant::now();
            let mut fp_map = self.build_state_fp_to_canon_fp(&registry);
            let witnesses = self.build_successor_witnesses(&registry, &mut fp_map);
            symmetry_fp_map_time = fp_map_start.elapsed();
            (Arc::new(fp_map), witnesses)
        } else {
            // No symmetry: defer state_cache build until check_liveness_property
            // needs it. state_fp_to_canon_fp is identity without symmetry —
            // consumers handle the empty-map case (check_property uses
            // unwrap_or(fp), safety_parts uses direct fp comparison).
            (Arc::new(FxHashMap::default()), None)
        };

        // Part of #2661: Use BFS-cached init states directly instead of O(N) scan.
        // Both full-state and fp-only modes record init states during BFS into
        // liveness_cache.init_states when cache_for_liveness is true (guaranteed
        // by the guard at line 138).
        let init_states_start = std::time::Instant::now();
        let init_state_fps: Vec<Fingerprint> = self
            .liveness_cache
            .init_states
            .iter()
            .map(|(fp, _)| *fp)
            .collect();
        let init_state_fps_time = init_states_start.elapsed();
        let init_state_count = self.liveness_cache.init_states.len();

        // Move `cached_successors` out temporarily to avoid borrow conflict:
        // `check_safety_temporal_property` needs `&mut self` but also reads successors.
        let cached_successors = std::mem::take(&mut self.liveness_cache.successors);

        if liveness_profile() {
            eprintln!("=== Liveness preparation ===");
            if state_cache_deferred {
                eprintln!(
                    "  state_cache build:   deferred ({} states pending)",
                    self.state_storage.seen.len()
                );
            } else {
                eprintln!(
                    "  state_cache build:   {:.3}s ({} states -> State{})",
                    state_cache_time.as_secs_f64(),
                    state_cache.len(),
                    if has_symmetry { ", symmetry" } else { "" }
                );
            }
            if has_symmetry {
                eprintln!(
                    "  fp map + witnesses:  {:.3}s ({} raw->canon mappings)",
                    symmetry_fp_map_time.as_secs_f64(),
                    state_fp_to_canon_fp.len()
                );
            }
            eprintln!(
                "  init_state_fps:      {:.3}s ({} init states)",
                init_state_fps_time.as_secs_f64(),
                init_state_count
            );
            eprintln!(
                "  cached_successors:   {} entries, {} total successor fps",
                cached_successors.len(),
                cached_successors.total_successors()
            );
            eprintln!(
                "  Total prep time:     {:.3}s",
                liveness_prep_start.elapsed().as_secs_f64()
            );
        }

        // Part of #3174: Bitmask-only mode. With tags < 64 enforced, all liveness
        // results are captured in the inline bitmask maps during BFS recording.
        // The per-tag cross-property caches are no longer needed.
        let inline_state_bitmasks = std::mem::take(&mut self.liveness_cache.inline_state_bitmasks);
        let inline_action_bitmasks =
            std::mem::take(&mut self.liveness_cache.inline_action_bitmasks);
        let liveness_result = self.run_liveness_properties(
            partial_graph,
            &registry,
            &init_state_fps,
            &cached_successors,
            succ_witnesses.as_ref(),
            &mut state_cache,
            &state_fp_to_canon_fp,
            &mut state_cache_deferred,
            &mut state_cache_time,
            &inline_state_bitmasks,
            &inline_action_bitmasks,
        );
        self.liveness_cache.inline_state_bitmasks = inline_state_bitmasks;
        self.liveness_cache.inline_action_bitmasks = inline_action_bitmasks;

        // Restore successor cache for post-check diagnostics / follow-on callers.
        self.liveness_cache.successors = cached_successors;
        liveness_result
    }
}
