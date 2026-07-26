// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! BFS/fingerprint-only exploration for grouped liveness plans.
//!
//! Extracted from `check_property.rs` (#3525) to keep both files under
//! the 500 LOC policy threshold.

use super::super::super::debug::liveness_profile;
use super::super::super::fingerprint::BfsFingerprintDomain;
use super::super::super::{
    check_error_to_result, Arc, ArrayState, CheckError, CheckResult, Fingerprint, LivenessChecker,
    ModelChecker, State,
};
use super::{GroupResolution, LivenessPropertyCtx, OtfExactRawCacheSession, OtfRetainedSuccessors};
use crate::error::EvalError;
use crate::liveness::GroupedLivenessPlan;
use crate::{ConfigCheckError, EvalCheckError, LivenessCheckError};
use rustc_hash::FxHashMap;
use tla_eval::tir::TirProgram;

impl ModelChecker<'_> {
    pub(in crate::check::model_checker::liveness) fn generate_liveness_successors_on_the_fly(
        &mut self,
        state: &State,
    ) -> Result<Vec<State>, CheckError> {
        let next_name = self
            .config
            .next
            .as_deref()
            .ok_or(ConfigCheckError::MissingNext)?;
        let successors = self.generate_successors(next_name, state)?;
        let registry = self.ctx.var_registry().clone();
        let current_arr = ArrayState::from_state(state, &registry);
        let mut filtered = Vec::with_capacity(successors.len() + 1);
        for successor in successors {
            let succ_arr = ArrayState::from_state(&successor, &registry);
            if self.check_state_constraints_array(&succ_arr)?
                && self.check_action_constraints_array(&current_arr, &succ_arr)?
            {
                filtered.push(successor);
            }
        }
        if self.exploration.stuttering_allowed {
            filtered.push(state.clone());
        }
        Ok(filtered)
    }

    /// Resolve one complete successor list from BFS-retained adjacency.
    ///
    /// Retained keys live in the frozen BFS fingerprint domain, while the
    /// on-the-fly behavior graph deliberately uses exact raw fingerprints.
    /// `array_state_fingerprint` performs the source-domain translation.
    /// Destination payloads are borrowed from the BFS-keyed wide-Init index; if even one is
    /// unavailable, return `None` so the caller evaluates Next for the whole
    /// source rather than constructing a partial edge set.
    pub(in crate::check::model_checker::liveness) fn replay_liveness_successors_from_retained(
        &mut self,
        state: &State,
        retained: &OtfRetainedSuccessors<'_>,
    ) -> Result<Option<Vec<State>>, CheckError> {
        let registry = self.ctx.var_registry().clone();
        let source_fp = match self.bfs_fingerprint_domain() {
            BfsFingerprintDomain::CompiledFlat => {
                let mut source_array = ArrayState::from_state(state, &registry);
                self.array_state_fingerprint(&mut source_array)?
            }
            BfsFingerprintDomain::FullStateFp64 | BfsFingerprintDomain::ArrayFp64
                if !self.nested_set_monitors_active() =>
            {
                let mut source_array = ArrayState::from_state(state, &registry);
                self.array_state_fingerprint(&mut source_array)?
            }
            BfsFingerprintDomain::FullStateFp64
            | BfsFingerprintDomain::ArrayFp64
            | BfsFingerprintDomain::View
            | BfsFingerprintDomain::SymmetryCanonical
            | BfsFingerprintDomain::FlatSymmetryCanonical => return Ok(None),
        };
        let Some(successor_fps) = retained.successor_fps(&source_fp) else {
            return Ok(None);
        };
        let mut successors = Vec::with_capacity(
            successor_fps.len() + usize::from(self.exploration.stuttering_allowed),
        );
        for successor_fp in successor_fps {
            let Some(successor) = retained.init_state(&successor_fp) else {
                return Ok(None);
            };
            successors.push(successor.to_state(&registry));
        }
        if self.exploration.stuttering_allowed {
            successors.push(state.clone());
        }
        Ok(Some(successors))
    }

    fn generate_liveness_successors_reusing_retained(
        &mut self,
        state: &State,
        retained: Option<&OtfRetainedSuccessors<'_>>,
    ) -> Result<(Vec<State>, bool), CheckError> {
        if let Some(retained) = retained {
            if let Some(successors) =
                self.replay_liveness_successors_from_retained(state, retained)?
            {
                return Ok((successors, true));
            }
        }
        self.generate_liveness_successors_on_the_fly(state)
            .map(|successors| (successors, false))
    }

    pub(in crate::check::model_checker::liveness) fn explore_grouped_liveness_plan_on_the_fly(
        &mut self,
        group_idx: usize,
        grouped_plan_count: usize,
        plan: &GroupedLivenessPlan,
        init_states: &[(Fingerprint, ArrayState)],
        retained_successors: Option<&OtfRetainedSuccessors<'_>>,
        tir: Option<&TirProgram>,
        exact_raw_cache: &mut OtfExactRawCacheSession,
        use_owned_compact_cache: bool,
    ) -> Result<LivenessChecker, CheckResult> {
        if liveness_profile() {
            eprintln!(
                "[liveness] group {}/{}: on-the-fly exploration ({} PEMs, {} check_state, {} check_action)...",
                group_idx + 1,
                grouped_plan_count,
                plan.pems.len(),
                plan.check_state.len(),
                plan.check_action.len()
            );
        }

        let tableau = crate::liveness::Tableau::new(plan.tf.clone());
        // Estimate behavior-graph node count as (discovered states * tableau nodes)
        // for auto-disk detection on large multi-property liveness specs.
        let estimated_nodes = self.stats.states_found.checked_mul(tableau.len().max(1));
        let mut checker = match LivenessChecker::new_from_env_with_hint(
            tableau,
            self.ctx.clone(),
            estimated_nodes,
        ) {
            Ok(checker) => checker,
            Err(error) => {
                return Err(check_error_to_result(
                    LivenessCheckError::RuntimeFailure(format!(
                        "Failed to create on-the-fly liveness checker for group {}: {error}",
                        group_idx + 1
                    ))
                    .into(),
                    &self.stats,
                ));
            }
        };
        checker.set_exact_raw_fp_leaf_fast_path_allowed(
            self.liveness_exact_raw_fp_leaf_fast_path_allowed(),
        );
        let stats = self.stats.clone();
        let needs_canonical_fp =
            self.compiled.cached_view_name.is_some() || !self.symmetry.perms.is_empty();
        debug_assert!(!use_owned_compact_cache || !needs_canonical_fp);
        let registry = self.ctx.var_registry().clone();
        let materialized_init_states = (!use_owned_compact_cache).then(|| {
            init_states
                .iter()
                .map(|(_, arr)| arr.to_state(&registry))
                .collect::<Vec<_>>()
        });
        if use_owned_compact_cache {
            // VIEW/symmetry need canonical fingerprints and concrete witnesses,
            // so only exact raw fingerprints use the owned compact cache.
            if let Some(cache) = exact_raw_cache.take() {
                checker.install_exact_raw_state_graph_cache(cache);
            } else {
                checker.enable_owned_behavior_graph_state_cache();
            }
        } else {
            // Do not retain an exact-raw cache while a canonical/legacy checker
            // allocates a separate state representation.
            exact_raw_cache.disable();
        }
        let checker_ref = std::cell::RefCell::new(self);
        let mut state_fp_to_canon_fp: Option<FxHashMap<Fingerprint, Fingerprint>> =
            needs_canonical_fp.then(FxHashMap::default);
        let explore_start = std::time::Instant::now();
        let mut retained_source_hits = 0usize;
        let mut retained_source_fallbacks = 0usize;
        let explore_result = {
            let mut get_successors = |state: &State| {
                let result = checker_ref
                    .borrow_mut()
                    .generate_liveness_successors_reusing_retained(state, retained_successors);
                match result {
                    Ok((successors, true)) => {
                        retained_source_hits += 1;
                        Ok(successors)
                    }
                    Ok((successors, false)) => {
                        if retained_successors.is_some() {
                            retained_source_fallbacks += 1;
                        }
                        Ok(successors)
                    }
                    Err(error) => Err(match error {
                        CheckError::Eval(EvalCheckError::Eval(inner)) => inner,
                        other => EvalError::Internal {
                            message: format!(
                                "on-the-fly liveness successor generation failed: {other}"
                            ),
                            span: None,
                        },
                    }),
                }
            };
            let mut state_fp_of = |state: &State| -> Result<Fingerprint, EvalError> {
                if let Some(fp_map) = state_fp_to_canon_fp.as_mut() {
                    let raw_fp = state.fingerprint();
                    if let Some(&canon_fp) = fp_map.get(&raw_fp) {
                        return Ok(canon_fp);
                    }
                    let canon_fp =
                        checker_ref
                            .borrow_mut()
                            .state_fingerprint(state)
                            .map_err(|error| match error {
                                CheckError::Eval(EvalCheckError::Eval(inner)) => inner,
                                other => EvalError::Internal {
                                    message: format!(
                                    "on-the-fly liveness fingerprint generation failed: {other}"
                                ),
                                    span: None,
                                },
                            })?;
                    fp_map.insert(raw_fp, canon_fp);
                    Ok(canon_fp)
                } else {
                    Ok(state.fingerprint())
                }
            };

            if matches!(&plan.tf, crate::liveness::LiveExpr::Bool(true)) {
                if use_owned_compact_cache {
                    checker.explore_state_graph_direct_with_raw_array_init_states(
                        init_states.iter().map(|(_, arr)| arr),
                        &registry,
                        &mut get_successors,
                    )
                } else {
                    checker.explore_state_graph_direct_with_state_fp(
                        materialized_init_states
                            .as_deref()
                            .expect("legacy direct exploration requires materialized roots"),
                        &mut get_successors,
                        &mut state_fp_of,
                    )
                }
            } else {
                if use_owned_compact_cache {
                    checker.explore_bfs_with_raw_array_init_states(
                        init_states.iter().map(|(_, arr)| arr),
                        &registry,
                        &mut get_successors,
                        tir,
                    )
                } else {
                    checker.explore_bfs_with_state_fp(
                        materialized_init_states
                            .as_deref()
                            .expect("legacy tableau exploration requires materialized roots"),
                        &mut get_successors,
                        tir,
                        &mut state_fp_of,
                    )
                }
            }
        };
        if let Err(error) = explore_result {
            return Err(check_error_to_result(
                EvalCheckError::Eval(error).into(),
                &stats,
            ));
        }
        if liveness_profile() && retained_successors.is_some() {
            eprintln!(
                "[liveness] retained adjacency: {retained_source_hits} source hits, \
                 {retained_source_fallbacks} Next fallbacks"
            );
        }
        if let Some(fp_map) = state_fp_to_canon_fp.filter(|map| !map.is_empty()) {
            checker.set_successor_maps(Arc::new(fp_map), None);
        }
        if liveness_profile() {
            // Part of #4083: log and collect thread-local cache stats before reporting.
            crate::liveness::log_cache_stats();
            checker.collect_cache_stats();
            let stats = checker.stats();
            eprintln!(
                "[liveness] group {}: on-the-fly explore {:.3}s ({} nodes, {} edges, {} checks)",
                group_idx + 1,
                explore_start.elapsed().as_secs_f64(),
                stats.graph_nodes,
                stats.graph_edges,
                stats.consistency_checks
            );
        }

        Ok(checker)
    }

    pub(in crate::check::model_checker::liveness) fn explore_grouped_liveness_plan(
        &mut self,
        group_idx: usize,
        grouped_plan_count: usize,
        plan: &GroupedLivenessPlan,
        ctx: &LivenessPropertyCtx<'_>,
        resolved: &GroupResolution,
        tir: Option<&TirProgram>,
    ) -> Result<LivenessChecker, CheckResult> {
        if liveness_profile() {
            eprintln!(
                "[liveness] group {}/{}: building tableau ({} PEMs, {} check_state, {} check_action)...",
                group_idx + 1,
                grouped_plan_count,
                plan.pems.len(),
                plan.check_state.len(),
                plan.check_action.len()
            );
        }
        let tableau_start = if liveness_profile() {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let tableau = crate::liveness::Tableau::new(plan.tf.clone());
        if let Some(start) = tableau_start {
            eprintln!(
                "[liveness] group {}: tableau built in {:.3}s ({} nodes, {} init)",
                group_idx + 1,
                start.elapsed().as_secs_f64(),
                tableau.len(),
                tableau.init_count()
            );
        } else if liveness_profile() {
            eprintln!(
                "[liveness] group {}: tableau has {} nodes, {} init",
                group_idx + 1,
                tableau.len(),
                tableau.init_count()
            );
        }

        let ctx_start = if liveness_profile() {
            Some(std::time::Instant::now())
        } else {
            None
        };
        // Estimate behavior-graph node count as (discovered states * tableau nodes)
        // for auto-disk detection on large multi-property liveness specs.
        // Use self.stats.states_found (full BFS count) rather than
        // ctx.cached_successors.len() which may be incomplete.
        let estimated_nodes = self.stats.states_found.checked_mul(tableau.len().max(1));
        let mut checker = match LivenessChecker::new_from_env_with_hint(
            tableau,
            self.ctx.clone(),
            estimated_nodes,
        ) {
            Ok(checker) => checker,
            Err(error) => {
                return Err(check_error_to_result(
                    LivenessCheckError::RuntimeFailure(format!(
                        "Failed to create liveness checker for group {}: {error}",
                        group_idx + 1
                    ))
                    .into(),
                    &self.stats,
                ));
            }
        };
        checker.set_exact_raw_fp_leaf_fast_path_allowed(
            self.liveness_exact_raw_fp_leaf_fast_path_allowed(),
        );
        if let Some(start) = ctx_start {
            eprintln!(
                "[liveness] group {}: checker created (ctx.clone) in {:.3}s",
                group_idx + 1,
                start.elapsed().as_secs_f64(),
            );
        }
        checker.set_successor_maps(
            Arc::clone(&resolved.state_fp_to_canon_fp),
            ctx.succ_witnesses.map(Arc::clone),
        );

        let add_stuttering = self.exploration.stuttering_allowed;
        let init_states = if resolved.no_tableau_fast_path {
            None
        } else {
            let init_state_materialize_start = std::time::Instant::now();
            let registry = self.ctx.var_registry().clone();
            let states: Vec<State> = self
                .liveness_cache
                .init_states
                .iter()
                .map(|(_, arr)| arr.to_state(&registry))
                .collect();
            if liveness_profile() {
                eprintln!(
                    "  init_states build:   {:.3}s ({} init states, tableau path)",
                    init_state_materialize_start.elapsed().as_secs_f64(),
                    states.len()
                );
            }
            Some(states)
        };
        let cached_successors = ctx.cached_successors;
        let group_state_cache = &resolved.state_cache;
        // Registry to reconstruct transient `State`s from the compact ArrayState
        // group cache for the explore closures (successors are consumed
        // immediately by explore_bfs; not retained).
        let registry = self.ctx.var_registry().clone();
        let group_state_fp_to_canon_fp = &resolved.state_fp_to_canon_fp;
        let mut state_fp_of = |state: &State| -> Result<Fingerprint, EvalError> {
            let raw_fp = state.fingerprint();
            Ok(group_state_fp_to_canon_fp
                .get(&raw_fp)
                .copied()
                .unwrap_or(raw_fp))
        };
        let mut get_successors = |state: &State| {
            let raw_fp = state.fingerprint();
            let fp = group_state_fp_to_canon_fp
                .get(&raw_fp)
                .copied()
                .unwrap_or(raw_fp);
            // Part of #4080: Use get_ref() to avoid cloning the entire Vec<Fingerprint>
            // on every lookup in the in-memory backend. Falls back to get() for disk.
            let owned_fallback;
            let succs_slice: Option<&[Fingerprint]> =
                if let Some(s) = cached_successors.get_ref(&fp) {
                    Some(s)
                } else {
                    owned_fallback = cached_successors.get(&fp);
                    owned_fallback.as_deref()
                };
            let mut succs: Vec<State> = succs_slice
                .map(|fps| {
                    fps.iter()
                        .filter_map(|sfp| {
                            group_state_cache
                                .get(sfp)
                                .map(|arr| arr.to_state(&registry))
                        })
                        .collect()
                })
                .unwrap_or_default();
            if add_stuttering && succs_slice.is_some() {
                succs.push(state.clone());
            }
            Ok(succs)
        };

        if liveness_profile() {
            eprintln!(
                "[liveness] group {}: starting {} ({} init states)...",
                group_idx + 1,
                if resolved.no_tableau_fast_path {
                    "explore_state_graph_direct_fp"
                } else {
                    "explore_bfs"
                },
                ctx.init_fps.len()
            );
        }
        let bfs_start = std::time::Instant::now();
        let explore_result = if resolved.no_tableau_fast_path {
            checker.set_behavior_graph_shared_cache(Arc::clone(&resolved.state_cache));
            let mut get_successor_fps = |fp: Fingerprint| -> Result<Vec<Fingerprint>, EvalError> {
                let canon_fp = group_state_fp_to_canon_fp.get(&fp).copied().unwrap_or(fp);
                let entry = cached_successors.get(&canon_fp);
                let has_entry = entry.is_some();
                let mut succs: Vec<Fingerprint> = entry.unwrap_or_default();
                if add_stuttering && has_entry {
                    succs.push(fp);
                }
                Ok(succs)
            };
            let result =
                checker.explore_state_graph_direct_fp(ctx.init_fps, &mut get_successor_fps);
            if let Err(error) = checker.populate_state_successor_fps_from_graph() {
                return Err(check_error_to_result(
                    LivenessCheckError::RuntimeFailure(format!(
                        "Failed to derive liveness successor fingerprints for group {}: {error}",
                        group_idx + 1
                    ))
                    .into(),
                    &self.stats,
                ));
            }
            result
        } else {
            // Flat-state win for the tableau path: share the compact ArrayState
            // group cache with the behavior graph so it reconstructs `State`
            // lazily (for trace/eval) instead of retaining a fresh `im::OrdMap`
            // per state, and so the explore path stores successor fingerprints
            // rather than `Vec<State>`. This is the SAME cache `get_successors`
            // already reads from, so completeness/soundness is unchanged.
            checker.set_behavior_graph_shared_cache(Arc::clone(&resolved.state_cache));
            checker.explore_bfs_with_state_fp(
                init_states
                    .as_deref()
                    .expect("tableau explore_bfs path must materialize init states"),
                &mut get_successors,
                tir,
                &mut state_fp_of,
            )
        };
        if let Err(error) = explore_result {
            return Err(check_error_to_result(
                EvalCheckError::Eval(error).into(),
                &self.stats,
            ));
        }

        if liveness_profile() {
            // Part of #4083: collect thread-local cache stats before reporting.
            checker.collect_cache_stats();
            let stats = checker.stats();
            eprintln!(
                "[liveness] group {}: explore_bfs {:.3}s ({} nodes, {} edges, {} checks)",
                group_idx + 1,
                bfs_start.elapsed().as_secs_f64(),
                stats.graph_nodes,
                stats.graph_edges,
                stats.consistency_checks
            );
            eprintln!(
                "=== Liveness profiling (group {}/{}) ===",
                group_idx + 1,
                grouped_plan_count
            );
            eprintln!(
                "  init_state_time:     {:.3}s",
                stats.init_state_time_us as f64 / 1_000_000.0
            );
            eprintln!(
                "  state_clone_time:    {:.3}s",
                stats.state_clone_time_us as f64 / 1_000_000.0
            );
            eprintln!(
                "  get_successors_time: {:.3}s",
                stats.get_successors_time_us as f64 / 1_000_000.0
            );
            eprintln!(
                "  add_successors_time: {:.3}s",
                stats.add_successors_time_us as f64 / 1_000_000.0
            );
            eprintln!("  consistency_checks:  {}", stats.consistency_checks);
            eprintln!("  graph_nodes:         {}", stats.graph_nodes);
            eprintln!("  graph_edges:         {}", stats.graph_edges);
            eprintln!("  pems_in_group:       {}", plan.pems.len());
            // Part of #4083: cache hit/miss statistics
            let sub_total = stats.subscript_cache_hits + stats.subscript_cache_misses;
            let sub_rate = if sub_total > 0 {
                stats.subscript_cache_hits as f64 / sub_total as f64 * 100.0
            } else {
                0.0
            };
            eprintln!(
                "  subscript_cache:     {} hits / {} misses ({:.1}%), {} evictions",
                stats.subscript_cache_hits,
                stats.subscript_cache_misses,
                sub_rate,
                stats.subscript_cache_evictions
            );
            let en_total = stats.enabled_cache_hits + stats.enabled_cache_misses;
            let en_rate = if en_total > 0 {
                stats.enabled_cache_hits as f64 / en_total as f64 * 100.0
            } else {
                0.0
            };
            eprintln!(
                "  enabled_cache:       {} hits / {} misses ({:.1}%), {} evictions",
                stats.enabled_cache_hits,
                stats.enabled_cache_misses,
                en_rate,
                stats.enabled_cache_evictions
            );
            let cc_total = stats.consistency_cache_hits + stats.consistency_cache_misses;
            let cc_rate = if cc_total > 0 {
                stats.consistency_cache_hits as f64 / cc_total as f64 * 100.0
            } else {
                0.0
            };
            eprintln!(
                "  consistency_cache:   {} hits / {} misses ({:.1}%)",
                stats.consistency_cache_hits, stats.consistency_cache_misses, cc_rate
            );
            let se_total = stats.state_env_cache_hits + stats.state_env_cache_misses;
            let se_rate = if se_total > 0 {
                stats.state_env_cache_hits as f64 / se_total as f64 * 100.0
            } else {
                0.0
            };
            eprintln!(
                "  state_env_cache:     {} hits / {} misses ({:.1}%)",
                stats.state_env_cache_hits, stats.state_env_cache_misses, se_rate
            );
            eprintln!("===================================");
        }

        Ok(checker)
    }
}
