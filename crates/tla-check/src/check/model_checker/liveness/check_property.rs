// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#[cfg(debug_assertions)]
use super::super::debug::debug_liveness_formula;
use super::super::debug::liveness_profile;
use super::super::{
    check_error_to_result, Arc, ArrayState, CheckResult, Expr, Fingerprint, FxHashMap, LiveExpr,
    LivenessChecker, ModelChecker, PropertySafetyParts, Spanned, State, SuccessorWitnessMap,
};
use crate::liveness::GroupedLivenessPlan;
use crate::state::{compute_fingerprint_from_compact_array, FpHashMap};
use crate::storage::{
    ActionBitmaskLookup, ActionBitmaskMap, StateBitmaskLookup, StateBitmaskMap, SuccessorGraph,
};
use crate::var_index::VarRegistry;
use crate::ConfigCheckError;
use tla_eval::tir::TirProgram;

mod exact_raw_cache;
mod explore;
mod fp_only;
mod results;
pub(super) use exact_raw_cache::OtfExactRawCacheSession;

/// Bundled context for checking a liveness property.
///
/// Groups the cached state/successor/bitmask data that flows from
/// `run_liveness_properties` through `check_liveness_property` and its
/// callees, reducing the argument count below clippy's `too_many_arguments`
/// threshold.
pub(in crate::check::model_checker) struct LivenessPropertyCtx<'a> {
    pub init_fps: &'a [Fingerprint],
    pub cached_successors: &'a SuccessorGraph,
    pub state_cache: &'a Arc<FxHashMap<Fingerprint, crate::state::ArrayState>>,
    pub state_fp_to_canon_fp: &'a Arc<FxHashMap<Fingerprint, Fingerprint>>,
    pub succ_witnesses: Option<&'a Arc<SuccessorWitnessMap>>,
    pub cross_state_bitmasks: &'a StateBitmaskMap,
    pub cross_action_bitmasks: &'a ActionBitmaskMap,
}

/// Run-scoped BFS adjacency retained across the automatic regeneration trip.
///
/// The graph is keyed in the frozen BFS fingerprint domain. Initial-state
/// tuples carry keys in that same domain, so wide-Init specs can resolve a
/// retained successor payload without duplicating the ArrayState itself. A
/// source is replayed only when every destination key resolves through this
/// index; partial resolution always falls back to evaluating Next. The session
/// owns the graph so both it and the Init index can be released between groups
/// once an admitted exact-raw cache covers the fixed run-wide roots.
pub(in crate::check::model_checker::liveness) struct OtfRetainedSuccessors<'a> {
    graph: Option<SuccessorGraph>,
    init_states: &'a [(Fingerprint, ArrayState)],
    init_index: Option<FpHashMap<usize>>,
}

impl<'a> OtfRetainedSuccessors<'a> {
    pub(in crate::check::model_checker::liveness) fn new(
        graph: SuccessorGraph,
        init_states: &'a [(Fingerprint, ArrayState)],
    ) -> Self {
        let mut init_index = crate::state::fp_hashmap_with_capacity(init_states.len());
        for (idx, (fp, _)) in init_states.iter().enumerate() {
            init_index.entry(*fp).or_insert(idx);
        }
        Self {
            graph: Some(graph),
            init_states,
            init_index: Some(init_index),
        }
    }

    fn successor_fps(&self, source_fp: &Fingerprint) -> Option<Vec<Fingerprint>> {
        self.graph.as_ref()?.get(source_fp)
    }

    fn init_state(&self, fp: &Fingerprint) -> Option<&ArrayState> {
        let idx = *self.init_index.as_ref()?.get(fp)?;
        let (stored_fp, state) = self.init_states.get(idx)?;
        debug_assert_eq!(stored_fp, fp);
        Some(state)
    }

    pub(in crate::check::model_checker::liveness) fn is_active(&self) -> bool {
        self.graph.is_some()
    }

    fn exact_raw_cache_floor_counts(&self, add_stuttering: bool) -> Option<(usize, usize, usize)> {
        let graph = self.graph.as_ref()?;
        let sources = graph.len();
        let successor_values = graph
            .total_successors()
            .saturating_add(add_stuttering.then_some(sources).unwrap_or_default());
        Some((sources, sources, successor_values))
    }

    /// Complete the checker-owned exact cache from retained adjacency when
    /// every retained parent and destination resolves through wide Init.
    ///
    /// The tuple keys are in the frozen BFS fingerprint domain. Exact-cache
    /// keys are recomputed from the authoritative compact values, so compiled
    /// or otherwise foreign tuple fingerprints are never reused as raw keys.
    /// Existing exact entries avoid disk reads; only tableau-pruned gaps read
    /// and translate their retained ordered successor lists.
    fn complete_exact_raw_cache_from_retained(
        &self,
        checker: &mut LivenessChecker,
        registry: &VarRegistry,
        add_stuttering: bool,
    ) -> bool {
        let started = std::time::Instant::now();
        let (Some(graph), Some(init_index)) = (&self.graph, &self.init_index) else {
            return false;
        };
        if graph.len() > init_index.len() {
            return false;
        }

        let mut covered_parents = 0usize;
        let mut translated_sources = 0usize;
        for (source_idx, (bfs_fp, source)) in self.init_states.iter().enumerate() {
            // The production Init slice is already deduplicated. Preserve the
            // constructor's first-writer behavior for defensive collision
            // fixtures without giving up sequential payload access.
            if init_index.get(bfs_fp).copied() != Some(source_idx) {
                continue;
            }
            if !graph.contains_parent(bfs_fp) {
                continue;
            }
            covered_parents += 1;

            let raw_fp = compute_fingerprint_from_compact_array(source.values(), registry);
            if checker.exact_raw_source_is_present_for(raw_fp, source) {
                continue;
            }

            let Some(successor_fps) = graph.get(bfs_fp) else {
                return false;
            };
            let mut successors = Vec::with_capacity(successor_fps.len());
            for successor_fp in &successor_fps {
                let Some(successor) = self.init_state(successor_fp) else {
                    return false;
                };
                successors.push(successor);
            }
            if !checker.seed_exact_raw_source_from_arrays(
                source,
                successors,
                registry,
                add_stuttering,
            ) || !checker.exact_raw_source_is_present_for(raw_fp, source)
            {
                return false;
            }
            translated_sources += 1;
        }

        let complete = covered_parents == graph.len();
        if complete && liveness_profile() {
            eprintln!(
                "[liveness] exact raw cache covers {covered_parents}/{} retained sources ({translated_sources} translated from retained adjacency) in {:.3}s",
                graph.len(),
                started.elapsed().as_secs_f64(),
            );
        }
        complete
    }

    /// Release graph-owned edge storage and the wide-Init fingerprint index.
    ///
    /// A complete exact-raw cache remains authoritative for every source the
    /// retained graph could replay. Sources that were never retained still
    /// evaluate Next because an inactive session returns `None`.
    pub(in crate::check::model_checker::liveness) fn release(&mut self) -> bool {
        let was_active = self.graph.take().is_some();
        self.init_index = None;
        was_active
    }

    pub(in crate::check::model_checker::liveness) fn into_graph(self) -> Option<SuccessorGraph> {
        self.graph
    }
}

/// Resolved per-group state cache from `resolve_group_state_cache`.
pub(super) struct GroupResolution {
    pub(super) state_cache: Arc<FxHashMap<Fingerprint, crate::state::ArrayState>>,
    pub(super) state_fp_to_canon_fp: Arc<FxHashMap<Fingerprint, Fingerprint>>,
    pub(super) no_tableau_fast_path: bool,
}

impl ModelChecker<'_> {
    pub(in crate::check::model_checker) fn liveness_exact_raw_fp_leaf_fast_path_allowed(
        &self,
    ) -> bool {
        !crate::tir_mode::tir_eval_stats_requested()
            && self
                .tir_parity
                .as_ref()
                .is_none_or(super::super::tir_parity::TirParityState::is_implicit_default_eval_mode)
    }

    fn property_definition_body(&self, prop_name: &str) -> Result<Spanned<Expr>, CheckResult> {
        self.module
            .op_defs
            .get(prop_name)
            .map(|def| def.body.clone())
            .ok_or_else(|| {
                check_error_to_result(
                    ConfigCheckError::MissingProperty(prop_name.to_string()).into(),
                    &self.stats,
                )
            })
    }

    fn separate_property_parts_with_profile(
        &mut self,
        prop_name: &str,
        body: &Spanned<Expr>,
    ) -> Option<(PropertySafetyParts, Option<Spanned<Expr>>)> {
        let split_start = if liveness_profile() {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let (safety_parts, liveness_expr) = self.separate_safety_liveness_parts(prop_name, body)?;
        if let Some(start) = split_start {
            eprintln!(
                "  separate_safety_liveness_parts: {:.3}s (init_terms={}, always_terms={})",
                start.elapsed().as_secs_f64(),
                safety_parts.init_terms.len(),
                safety_parts.always_terms.len(),
            );
        }
        Some((safety_parts, liveness_expr))
    }

    fn check_property_safety_parts_with_profile(
        &mut self,
        prop_name: &str,
        safety_parts: &PropertySafetyParts,
        cached_successors: &SuccessorGraph,
        succ_witnesses: Option<&Arc<SuccessorWitnessMap>>,
    ) -> Option<CheckResult> {
        let safety_start = if liveness_profile() {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let init_state_cache = self.liveness_cache.init_states.clone();
        let result = self.check_property_safety_parts(
            prop_name,
            safety_parts,
            &init_state_cache,
            cached_successors,
            succ_witnesses,
        );
        if let Some(start) = safety_start {
            let transition_count = cached_successors.total_successors();
            eprintln!(
                "  check_property_safety_parts: {:.3}s (transitions={})",
                start.elapsed().as_secs_f64(),
                transition_count,
            );
        }
        result
    }

    fn resolve_grouped_plans(
        &mut self,
        prop_name: &str,
        liveness_expr: &Spanned<Expr>,
    ) -> Result<(Vec<GroupedLivenessPlan>, u32), CheckResult> {
        if let Some(plan) = self.inline_property_plan(prop_name) {
            if liveness_profile() {
                let dnf_clause_count: usize = plan.grouped_plans.iter().map(|p| p.pems.len()).sum();
                eprintln!(
                    "[liveness] reusing inline grouped plans: {} groups, {} DNF clauses total",
                    plan.grouped_plans.len(),
                    dnf_clause_count
                );
            }
            Ok((plan.grouped_plans.clone(), plan.max_fairness_tag))
        } else {
            self.build_grouped_plans_for_property(prop_name, liveness_expr)
        }
    }

    fn resolve_group_state_cache(
        &mut self,
        prop_name: &str,
        plan: &GroupedLivenessPlan,
        max_fairness_tag: u32,
        ctx: &LivenessPropertyCtx<'_>,
    ) -> Result<GroupResolution, CheckResult> {
        let has_inline_results = self.inline_property_plan(prop_name).is_some()
            || !ctx.cross_state_bitmasks.is_empty()
            || !ctx.cross_action_bitmasks.is_empty();
        let max_inline_tag = self
            .inline_property_plan(prop_name)
            .map_or(max_fairness_tag, |plan| plan.max_cached_tag);
        // #4159 follow-up: whether the inline bitmask backend can serve the full
        // multi-word LiveBitmask (tags >= 64). When it can, >63-tag leaves are
        // reconstructable on the fast path; otherwise we stay fail-closed at tag < 64.
        let multiword = if let Some(prop_plan) = self.inline_property_plan(prop_name) {
            prop_plan.inline_results().multiword_capable()
        } else {
            ctx.cross_state_bitmasks.multiword_capable()
                && ctx.cross_action_bitmasks.multiword_capable()
        };
        let no_tableau_fast_path = matches!(&plan.tf, LiveExpr::Bool(true));
        let needs_full_state_cache = match self.liveness_mode {
            super::LivenessMode::FingerprintOnly { .. } => {
                !no_tableau_fast_path
                    || !fp_only::all_checks_structurally_cached(
                        plan,
                        max_fairness_tag,
                        max_inline_tag,
                        has_inline_results,
                        multiword,
                    )
            }
            super::LivenessMode::Disabled | super::LivenessMode::FullState { .. } => false,
        };

        if needs_full_state_cache {
            let (state_cache, state_fp_to_canon_fp) =
                self.build_fp_only_liveness_state_cache(ctx.init_fps, ctx.cached_successors)?;
            Ok(GroupResolution {
                state_cache,
                state_fp_to_canon_fp,
                no_tableau_fast_path,
            })
        } else {
            Ok(GroupResolution {
                state_cache: Arc::clone(ctx.state_cache),
                state_fp_to_canon_fp: Arc::clone(ctx.state_fp_to_canon_fp),
                no_tableau_fast_path,
            })
        }
    }

    /// Run one grouped liveness check pass with the property's inline bitmask
    /// caches installed. Split out of `check_grouped_liveness_plan` so the
    /// `CandidateStatesUnavailable` retry can re-run it after materializing the
    /// fp-only state cache (the inline-results borrow of `self` must end before
    /// `build_fp_only_liveness_state_cache(&mut self)` can run).
    fn run_grouped_liveness_check_pass(
        &self,
        prop_name: &str,
        plan: &GroupedLivenessPlan,
        max_fairness_tag: u32,
        ctx: &LivenessPropertyCtx<'_>,
        tir: Option<&TirProgram>,
        checker: &mut LivenessChecker,
    ) -> crate::liveness::LivenessResult {
        let inline_results = if let Some(prop_plan) = self.inline_property_plan(prop_name) {
            Some(prop_plan.inline_results())
        } else if !ctx.cross_state_bitmasks.is_empty() || !ctx.cross_action_bitmasks.is_empty() {
            Some(crate::liveness::InlineCheckResults {
                max_tag: max_fairness_tag,
                state_bitmasks: ctx.cross_state_bitmasks,
                action_bitmasks: ctx.cross_action_bitmasks,
            })
        } else {
            None
        };
        checker.check_liveness_grouped_with_inline_cache(
            plan,
            max_fairness_tag,
            inline_results,
            tir,
        )
    }

    fn check_grouped_liveness_plan(
        &mut self,
        prop_name: &str,
        group_idx: usize,
        plan: &GroupedLivenessPlan,
        max_fairness_tag: u32,
        ctx: &LivenessPropertyCtx<'_>,
        tir: Option<&TirProgram>,
        checker: &mut LivenessChecker,
    ) -> Option<CheckResult> {
        if liveness_profile() {
            eprintln!(
                "[liveness] group {}: starting SCC + PEM checking...",
                group_idx + 1
            );
        }
        let check_start = std::time::Instant::now();
        let mut check_result = self.run_grouped_liveness_check_pass(
            prop_name,
            plan,
            max_fairness_tag,
            ctx,
            tir,
            checker,
        );

        // SOUNDNESS (#liveness-fp-only-false-hold): a candidate violating cycle
        // was found, but the fairness re-verification gate could not materialize
        // the cycle's concrete states (fingerprint-only fast path keeps no state
        // cache). Materialize the full fp-only replay cache (cached across
        // properties), install it on the behavior graph, and re-run the check so
        // the gate can authoritatively confirm or refute the witness. Without
        // this retry, the old behavior silently refuted EVERY witness in this
        // mode — reporting genuine violations as HOLDs.
        if let crate::liveness::LivenessResult::CandidateStatesUnavailable { missing_fp } =
            check_result
        {
            if liveness_profile() {
                eprintln!(
                    "[liveness] group {}: candidate cycle state {} unavailable — \
                     materializing fp-only state cache and re-running",
                    group_idx + 1,
                    missing_fp
                );
            }
            // A replay cache built during a PERIODIC (partial-graph) liveness run
            // can be stale — missing states explored after it was built. If the
            // cached map doesn't cover the fingerprint the gate needs, drop it so
            // the rebuild below replays the now-complete successor graph.
            if self
                .liveness_cache
                .fp_only_replay_cache
                .as_ref()
                .is_some_and(|(cache, _)| !cache.contains_key(&missing_fp))
            {
                self.liveness_cache.fp_only_replay_cache = None;
            }
            let (state_cache, _state_fp_to_canon_fp) = match self
                .build_fp_only_liveness_state_cache(ctx.init_fps, ctx.cached_successors)
            {
                Ok(result) => result,
                Err(check_result) => return Some(check_result),
            };
            checker.set_behavior_graph_shared_cache(state_cache);
            check_result = self.run_grouped_liveness_check_pass(
                prop_name,
                plan,
                max_fairness_tag,
                ctx,
                tir,
                checker,
            );
        }
        if liveness_profile() {
            crate::liveness::log_cache_stats();
        }
        checker.collect_cache_stats();
        if liveness_profile() {
            let stats = checker.stats();
            eprintln!(
                "[liveness cache] consistency: hits={}, misses={}",
                stats.consistency_cache_hits, stats.consistency_cache_misses
            );
            eprintln!(
                "[liveness cache] state_env: hits={}, misses={}",
                stats.state_env_cache_hits, stats.state_env_cache_misses
            );
            eprintln!(
                "[liveness] group {}: SCC done in {:.3}s",
                group_idx + 1,
                check_start.elapsed().as_secs_f64()
            );
            eprintln!(
                "  check_liveness_grouped time: {:.3}s",
                check_start.elapsed().as_secs_f64()
            );
        }
        self.map_liveness_result(prop_name, check_result)
    }

    /// Check a single liveness property
    ///
    /// Returns `Some(CheckResult)` if the property is violated, `None` if satisfied.
    ///
    /// Part of #3065: `state_cache` is `Arc`-wrapped for zero-copy sharing with
    /// the behavior graph on the fingerprint-based direct path. `init_fps` provides
    /// initial state fingerprints for the same path.
    pub(in crate::check::model_checker) fn check_liveness_property(
        &mut self,
        prop_name: &str,
        ctx: &LivenessPropertyCtx<'_>,
    ) -> Option<CheckResult> {
        let func_start = std::time::Instant::now();

        crate::liveness::clear_enabled_cache();
        crate::liveness::clear_leaf_result_cache();
        self.rearm_inline_fairness_metadata();
        if liveness_profile() {
            eprintln!("[liveness] check_liveness_property: starting '{prop_name}'");
        }

        let body = match self.property_definition_body(prop_name) {
            Ok(body) => body,
            Err(result) => return Some(result),
        };
        let (safety_parts, liveness_expr) =
            self.separate_property_parts_with_profile(prop_name, &body)?;
        if let Some(result) = self.check_property_safety_parts_with_profile(
            prop_name,
            &safety_parts,
            ctx.cached_successors,
            ctx.succ_witnesses,
        ) {
            return Some(result);
        }

        let liveness_expr = liveness_expr?;
        let (grouped_plans, max_fairness_tag) =
            match self.resolve_grouped_plans(prop_name, &liveness_expr) {
                Ok(result) => result,
                Err(check_result) => return Some(check_result),
            };

        let tir_modules = self
            .tir_parity
            .as_ref()
            .and_then(|tp| tp.clone_modules_for_selected_eval(prop_name));
        let tir = tir_modules.as_ref().map(|(root, deps)| {
            let dep_refs: Vec<&_> = deps.iter().collect();
            TirProgram::from_modules(root, &dep_refs)
        });

        for (group_idx, plan) in grouped_plans.iter().enumerate() {
            debug_eprintln!(
                debug_liveness_formula(),
                "[DEBUG] Starting grouped plan {} ({} PEMs)",
                group_idx,
                plan.pems.len()
            );

            let resolved =
                match self.resolve_group_state_cache(prop_name, plan, max_fairness_tag, ctx) {
                    Ok(result) => result,
                    Err(check_result) => return Some(check_result),
                };
            let mut checker = match self.explore_grouped_liveness_plan(
                group_idx,
                grouped_plans.len(),
                plan,
                ctx,
                &resolved,
                tir.as_ref(),
            ) {
                Ok(checker) => checker,
                Err(check_result) => return Some(check_result),
            };
            if let Some(result) = self.check_grouped_liveness_plan(
                prop_name,
                group_idx,
                plan,
                max_fairness_tag,
                ctx,
                tir.as_ref(),
                &mut checker,
            ) {
                return Some(result);
            }
        }

        if liveness_profile() {
            eprintln!(
                "Total check_liveness_property time: {:.3}s",
                func_start.elapsed().as_secs_f64()
            );
        }
        None
    }

    pub(in crate::check::model_checker::liveness) fn check_liveness_property_on_the_fly(
        &mut self,
        prop_name: &str,
        init_states: &[(Fingerprint, ArrayState)],
        mut retained_successors: Option<&mut OtfRetainedSuccessors<'_>>,
        exact_raw_cache: &mut OtfExactRawCacheSession,
        use_owned_compact_cache: bool,
    ) -> Option<CheckResult> {
        crate::liveness::clear_enabled_cache();
        crate::liveness::clear_leaf_result_cache();
        self.rearm_inline_fairness_metadata();
        if liveness_profile() {
            eprintln!("[liveness] on-the-fly check_liveness_property: starting '{prop_name}'");
        }

        let body = match self.property_definition_body(prop_name) {
            Ok(body) => body,
            Err(result) => return Some(result),
        };
        let (safety_parts, liveness_expr) =
            self.separate_property_parts_with_profile(prop_name, &body)?;
        if let Some(result) =
            self.check_property_safety_parts_on_the_fly(prop_name, &safety_parts, init_states)
        {
            return Some(result);
        }

        let liveness_expr = liveness_expr?;
        let (grouped_plans, max_fairness_tag) =
            match self.build_grouped_plans_for_property(prop_name, &liveness_expr) {
                Ok(result) => result,
                Err(check_result) => return Some(check_result),
            };

        let tir_modules = self
            .tir_parity
            .as_ref()
            .and_then(|tp| tp.clone_modules_for_selected_eval(prop_name));
        let tir = tir_modules.as_ref().map(|(root, deps)| {
            let dep_refs: Vec<&_> = deps.iter().collect();
            TirProgram::from_modules(root, &dep_refs)
        });

        for (group_idx, plan) in grouped_plans.iter().enumerate() {
            let retained_for_explore = retained_successors
                .as_deref()
                .filter(|retained| retained.is_active());
            let mut checker = match self.explore_grouped_liveness_plan_on_the_fly(
                group_idx,
                grouped_plans.len(),
                plan,
                init_states,
                retained_for_explore,
                tir.as_ref(),
                exact_raw_cache,
                use_owned_compact_cache,
            ) {
                Ok(checker) => checker,
                Err(check_result) => return Some(check_result),
            };
            let direct_traversal = matches!(&plan.tf, crate::liveness::LiveExpr::Bool(true));
            let retained_replacement_complete =
                retained_successors.as_deref().is_some_and(|retained| {
                    if !retained.is_active() || !exact_raw_cache.may_attempt_retained_release() {
                        false
                    } else {
                        retained
                            .exact_raw_cache_floor_counts(self.exploration.stuttering_allowed)
                            .is_some_and(|(states, sources, successor_values)| {
                                exact_raw_cache.retained_translation_floor_is_admitted(
                                    states,
                                    sources,
                                    successor_values,
                                    self.module.vars.len(),
                                ) && (direct_traversal
                                    || retained.complete_exact_raw_cache_from_retained(
                                        &mut checker,
                                        self.ctx.var_registry(),
                                        self.exploration.stuttering_allowed,
                                    ))
                            })
                    }
                });
            if exact_raw_cache.can_release_retained_before_check(
                &checker,
                self.module.vars.len(),
                retained_replacement_complete,
            ) {
                if let Some(retained) = retained_successors.as_deref_mut() {
                    if retained.release() && liveness_profile() {
                        eprintln!(
                            "[liveness] released retained BFS adjacency after complete \
                             exact-raw cache"
                        );
                    }
                }
            }
            // Exploration and any retained-adjacency translation are now
            // complete. Release redundant retained adjacency before packing so
            // the CSR copy does not overlap it at the high-water mark. Pack
            // only a structurally closed exact-raw relation; a later group can
            // append newly reachable rows through the sparse extension path.
            checker.freeze_complete_exact_raw_adjacency();
            let check_result = checker.check_liveness_grouped_with_inline_cache(
                plan,
                max_fairness_tag,
                None,
                tir.as_ref(),
            );
            exact_raw_cache.recover_from(&mut checker, self.module.vars.len());
            if liveness_profile() {
                crate::liveness::log_cache_stats();
            }
            checker.collect_cache_stats();
            if liveness_profile() {
                let stats = checker.stats();
                eprintln!(
                    "[liveness cache] consistency: hits={}, misses={}",
                    stats.consistency_cache_hits, stats.consistency_cache_misses
                );
                eprintln!(
                    "[liveness cache] state_env: hits={}, misses={}",
                    stats.state_env_cache_hits, stats.state_env_cache_misses
                );
            }
            if let Some(result) = self.map_liveness_result(prop_name, check_result) {
                return Some(result);
            }
        }

        None
    }
}
