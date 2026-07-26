// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Constructor implementation for `ModelChecker`.
//!
//! Extracted from `setup.rs` as part of #2359 Phase 2 decomposition.
//! Uses the shared `checker_setup::setup_checker_modules` pipeline (#810)
//! to perform module resolution, cfg rewrites, and operator/variable collection.

#[cfg(debug_assertions)]
use super::debug::debug_lazy_values_in_state_log_limit;
use super::debug::{
    debug_internal_fp_collision, debug_internal_fp_collision_limit, debug_seen_tlc_fp_dedup,
    debug_seen_tlc_fp_dedup_collision_limit, skip_liveness,
};
use super::{
    Arc, CapacityStatus, CheckpointState, CompiledSpec, CoverageState, DebugDiagnostics, Duration,
    ExplorationControl, FxHashMap, LivenessCacheState, LivenessMode, ModelChecker, Module,
    ModuleState, PeriodicLivenessState, RuntimeHooksState, StateStorage, SymmetryState,
    TraceLocationsStorage, TraceState,
};
use crate::check::model_checker::run_helpers::JIT_INITIAL_VALIDATION_COUNT;
use crate::check::model_checker::tir_parity::TirParityState;
use crate::check::model_checker::trust_cg_dispatch::TrustCgActionDispatchStats;
use crate::checker_setup::{setup_checker_modules, CheckerSetup, SetupOptions};
use crate::state::fp_hashmap;
use crate::storage::factory::{FingerprintSetFactory, StorageConfig};
use crate::storage::{FingerprintPayloadWitnesses, SuccessorGraph};
use crate::Config;
// Part of #4398: consume fail-closed compiled-backend types through tla-check's local shim.
use crate::compiled_backend_unavailable::RecompilationController as RecompilationControllerImpl;

/// Stage 4 of the unified-backend migration: `ModelChecker` exposes the exact facts the
/// compiled-BFS admission gate reads, so the decision can live in
/// `tla_backend::admit_compiled_bfs` (one auditable place). Pure delegation to existing
/// inherent methods/fields — inherent methods take precedence over the same-named trait
/// methods in the bodies below, so there is no recursion.
impl<'a> tla_backend::CompiledBfsFacts for ModelChecker<'a> {
    fn use_compiled_bfs_override(&self) -> Option<bool> {
        self.config.use_compiled_bfs
    }
    fn compiled_bfs_env_disabled(&self) -> bool {
        crate::check::debug::compiled_bfs_disabled()
    }
    fn flat_frontier_admitted(&self) -> bool {
        self.compiled_bfs_flat_frontier_admitted()
    }
    fn step_width_matches_flat_frontier(&self) -> bool {
        self.compiled_bfs_step_width_matches_flat_frontier()
    }
    fn implied_actions_require_interpreter_eval(&self) -> bool {
        // inherent method (precedence over this trait method) — no recursion.
        self.implied_actions_require_interpreter_eval()
    }
    fn step_evaluates_interpreter_implied_actions(&self) -> bool {
        self.compiled_bfs_step_evaluates_interpreter_implied_actions()
    }
    fn has_action_constraints(&self) -> bool {
        !self.config.action_constraints.is_empty()
    }
    fn coverage_collect(&self) -> bool {
        self.coverage.collect
    }
    fn has_state_constraints(&self) -> bool {
        !self.config.constraints.is_empty()
    }
    fn state_constrained_native_fused_admission_active(&self) -> bool {
        // inherent method (precedence over this trait method) — no recursion.
        self.state_constrained_native_fused_admission_active()
    }
    fn compiled_step_built(&self) -> bool {
        self.compiled_bfs_step.is_some()
    }
    fn compiled_level_built(&self) -> bool {
        self.compiled_bfs_level.is_some()
    }
    fn fully_flat_layout(&self) -> bool {
        self.flat_bfs_adapter
            .as_ref()
            .is_some_and(|a| a.is_fully_flat())
    }
}

impl<'a> ModelChecker<'a> {
    /// Whether the AUTO selector has structurally vetoed trust-cg for THIS checker
    /// instance (Stage 3: per-instance, replaces the process-global static).
    pub(in crate::check) fn trust_cg_structurally_vetoed(&self) -> bool {
        self.trust_cg_structural_veto
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Latch the AUTO-selector structural veto for this run: route the rest of the run
    /// to the interpreter. Idempotent. Only meaningful in AUTO mode; explicit
    /// `--backend trust-cg` never calls this, so harness/oracle runs keep forced-native.
    pub(in crate::check) fn set_trust_cg_structural_veto(&self) {
        self.trust_cg_structural_veto
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub(super) fn new_with_extends_impl(
        module: &'a Module,
        extended_modules: &[&Module],
        config: &'a Config,
    ) -> Self {
        // Use the shared setup pipeline (Part of #810).
        let CheckerSetup {
            mut ctx,
            main_module,
            rewritten_exts,
            unqualified_modules: _,
            vars,
            op_defs,
            assumes,
            setup_error,
        } = setup_checker_modules(
            module,
            extended_modules,
            config,
            &SetupOptions {
                load_instances: true,
            },
        );

        // MC-specific: resolve state variable references in operators already loaded
        // in the EvalCtx. The shared pipeline registers vars and resolves op_defs,
        // but ctx-internal ops need a separate resolution pass.
        ctx.resolve_state_vars_in_loaded_ops();

        // Part of #4053: Determine if the spec can produce lazy values at runtime.
        // When false, the per-successor `has_lazy_state_value` scan is skipped.
        let spec_may_produce_lazy =
            crate::materialize::spec_may_produce_lazy_values(module, extended_modules);

        // Symmetry permutations will be computed lazily after constants are bound
        let symmetry_perms = Vec::new();
        let symmetry_mvperms = Vec::new();

        // Map FileId -> module name for TLC-style location rendering.
        let root_module_name = module.name.node.clone();
        let mut file_id_to_module_name: FxHashMap<tla_core::FileId, String> = FxHashMap::default();
        file_id_to_module_name.insert(module.name.span.file, root_module_name.clone());
        for ext_mod in extended_modules {
            file_id_to_module_name
                .entry(ext_mod.name.span.file)
                .or_insert_with(|| ext_mod.name.node.clone());
        }

        let invariant_verdict_cache =
            super::super::invariants::InvariantVerdictCache::new(&ctx, &config.invariants);
        let state_constraint_verdict_cache =
            super::super::invariants::StateConstraintVerdictCache::new(&ctx, &config.constraints);

        ModelChecker {
            trust_cg_structural_veto: std::sync::atomic::AtomicBool::new(false),
            config,
            module: ModuleState {
                root_name: root_module_name,
                file_id_to_name: file_id_to_module_name,
                file_id_to_path: FxHashMap::default(),
                frontend_source_is_quint: false,
                setup_error,
                assumes,
                vars,
                op_defs,
            },
            ctx,
            state_storage: StateStorage {
                seen: fp_hashmap(),
                seen_fps: FingerprintSetFactory::create(StorageConfig::default())
                    .expect("in-memory storage creation is infallible"),
                retired_seen_fps_len: None,
                store_full_states: false,
                compiled_flat_payload_witnesses: FingerprintPayloadWitnesses::new(),
            },
            trace: TraceState {
                depths: fp_hashmap(),
                auto_create_trace_file: true,
                trace_file: None,
                trace_locs: TraceLocationsStorage::in_memory(),
                trace_degraded: false,
                current_parent_trace_loc: None,
                last_inserted_trace_loc: 0,
                lazy_trace_index: false,
                cached_init_name: None,
                cached_next_name: None,
                cached_resolved_next_name: None,
            },
            compiled: CompiledSpec {
                split_action_meta: None,
                split_action_complete_bindings: None,
                cached_view_name: None,
                uses_trace: false,
                promoted_property_invariants: Vec::new(),
                state_property_violation_names: Vec::new(),
                eval_implied_actions: Vec::new(),
                native_implied_actions: Vec::new(),
                eval_state_invariants: Vec::new(),
                promoted_implied_action_properties: Vec::new(),
                property_init_predicates: Vec::new(),
                action_constraint_analysis: None,
                state_constraints_reusable_on_exact_duplicate: false,
                pc_dispatch: None,
                pc_var_idx: None,
                spec_may_produce_lazy,
            },
            exploration: ExplorationControl {
                check_deadlock: config.check_deadlock,
                force_explicit_bfs: false,
                stuttering_allowed: true,
                continue_on_error: false,
                first_violation: None,
                first_action_property_violation: None,
                max_states: None,
                max_depth: None,
                deadline: None,
                memory_policy: None,
                disk_limit_bytes: None,
                internal_memory_limit: None,
            },
            stats: super::CheckStats::default(),
            run_diagnostics: std::sync::Arc::new(crate::run_diagnostics::RunDiagnostics::default()),
            hooks: RuntimeHooksState {
                init_progress_callback: None,
                progress_callback: None,
                progress_interval: 1000,
                last_capacity_status: CapacityStatus::Normal,
                last_memory_check: std::time::Instant::now(),
            },
            liveness_cache: LivenessCacheState {
                // Part of #3176: use disk-backed successor graph when env var is set.
                successors: if crate::liveness::debug::use_disk_successors() {
                    SuccessorGraph::disk().expect("disk successor graph creation failed")
                } else {
                    SuccessorGraph::default()
                },
                successor_witnesses: fp_hashmap(),
                witness_intern: fp_hashmap(),
                fairness: Vec::new(),
                fairness_state_checks: Vec::new(),
                fairness_action_checks: Vec::new(),
                fairness_max_tag: 0,
                action_provenance_tags: Vec::new(),
                action_fast_path_provenance_tags: Vec::new(),
                enabled_action_groups: Vec::new(),
                whole_next_enabled_tags: Vec::new(),
                whole_next_action_tags: Vec::new(),
                provenance_covered_tags: rustc_hash::FxHashSet::default(),
                provenance_uncovered_action_leaves: Vec::new(),
                enabled_provenance: Vec::new(),
                subscript_action_pairs: Vec::new(),
                // Part of #3177: use disk-backed bitmask maps when env var is set.
                inline_state_bitmasks: if crate::liveness::debug::use_disk_bitmasks() {
                    crate::storage::StateBitmaskMap::disk()
                        .expect("disk state bitmask map creation failed")
                } else {
                    crate::storage::StateBitmaskMap::default()
                },
                inline_action_bitmasks: if crate::liveness::debug::use_disk_bitmasks() {
                    crate::storage::ActionBitmaskMap::disk()
                        .expect("disk action bitmask map creation failed")
                } else {
                    crate::storage::ActionBitmaskMap::default()
                },
                inline_property_plans: Vec::new(),
                // Part of #3709: on-the-fly liveness regenerates successors
                // after BFS and therefore does not need the cached system graph.
                cache_for_liveness: !config.properties.is_empty()
                    && !skip_liveness()
                    && !config.liveness_execution.uses_on_the_fly(),
                #[cfg(feature = "testing")]
                force_capture_for_testing: false,
                init_states: Vec::new(),
                fp_only_replay_cache: None,
                bfs_seeded_states: rustc_hash::FxHashMap::default(),
                regenerate_on_the_fly: false,
            },
            liveness_mode: LivenessMode::compute(
                !config.properties.is_empty(),
                false,
                false,
                false,
            ),
            coverage: CoverageState {
                // Default off so the primary BFS routing / fingerprint domain is
                // unchanged. V2 dead-action tracking is enabled explicitly via
                // `set_collect_coverage` (the CLI turns it on by default so the
                // V2 WARNING is default-on; see TRUST_VACUITY_GATE §1.A).
                collect: false,
                display: false,
                default_dead_action_tracking: false,
                native_fast_path_skipped: false,
                actions: Arc::new(Vec::new()),
                retired_actions: Vec::new(),
                coverage_guided: false,
                tracker: None,
                mix_ratio: 8,
            },
            symmetry: SymmetryState {
                perms: symmetry_perms,
                mvperms: symmetry_mvperms,
                fp_cache: FxHashMap::default(),
                fp_cache_hits: 0,
                fp_cache_misses: 0,
                states_folded: 0,
                fp_cache_evictions: 0,
                group_names: Vec::new(),
                auto_detected: false,
                auto_symmetry_override: None,
                yield_cutoff_active: false,
                yield_cutoff_declined: false,
                yield_cutoff_fired_at: 0,
                yield_cutoff_skipped: 0,
                yield_merges: 0,
                yield_merge_probe: rustc_hash::FxHashSet::default(),
            },
            tir_parity: TirParityState::from_env(main_module, rewritten_exts),
            invariant_verdict_cache,
            state_constraint_verdict_cache,
            // Part of #3578: bytecode VM compilation happens after setup in
            // compile_invariant_bytecode(). Initialized None, populated lazily.
            bytecode: None,
            action_bytecode: None,
            value_action_vm: super::super::value_action_vm::ValueActionVmDispatch::from_env(),
            executed_tier_compiled: None,
            engine_tier_hot_swapped: false,
            // Lever 1 (#EWD998PCal): populated by compile_constraint_bytecode()
            // during run_prepare when CONSTRAINTs are present.
            constraint_bytecode: None,
            constraint_bytecode_disabled: false,
            // Part of #3582: JIT compilation happens after bytecode compilation.
            jit_cache: None,
            jit_state_scratch: Vec::new(),
            jit_all_compiled: false,
            jit_resolved_fns: None,
            jit_hits: 0,
            jit_misses: 0,
            jit_hit: 0,
            jit_fallback: 0,
            jit_not_compiled: 0,
            total_invariant_evals: 0,
            jit_verify_checked: 0,
            jit_verify_mismatches: 0,
            // Part of #3850: tiered JIT manager initialized in prepare_bfs_common
            // after action splitting discovers the action count.
            tier_manager: None,
            action_eval_counts: Vec::new(),
            action_succ_totals: Vec::new(),
            tier_promotion_history: Vec::new(),
            type_profile_scratch: Vec::new(),
            jit_next_state_cache: None,
            next_state_dispatch: tla_jit_abi::NextStateDispatchCounters::default(),
            jit_cache_build_stats: None,
            pending_jit_compilation: None,
            recompilation_controller: RecompilationControllerImpl::new(),
            jit_state_layout: None,
            jit_monolithic_disabled: false,
            jit_disabled_actions: Vec::new(),
            jit_all_next_state_compiled: false,
            jit_has_any_promoted: false,
            jit_state_all_scalar: false,
            jit_compiled_fp_active: false,
            #[cfg(debug_assertions)]
            fp_algorithm_sealed: false,
            frozen_bfs_fingerprint_domain: None,
            flat_fp_scratch: Vec::new(),
            jit_validation_remaining: JIT_INITIAL_VALIDATION_COUNT,
            jit_action_lookup_keys: Vec::new(),
            jit_inner_exists_keys: Vec::new(),
            jit_action_out_scratch: Vec::new(),
            jit_perf_monitor: (0, 0, 0),
            jit_diag_enabled: {
                static JIT_DIAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                *JIT_DIAG.get_or_init(|| std::env::var("TY_JIT_DIAG").is_ok())
            },
            compiled_bfs_step: None,
            compiled_bfs_level: None,
            deferred_fused_level_build: false,
            deferred_fused_invariant_build: false,
            trust_cg_lazy_pending: false,
            // Part of #4118: trust-codegen native compilation cache.
            trust_cg_cache: None,
            trust_cg_build_stats: None,
            setup_trace: None,
            trust_cg_action_dispatch_stats: TrustCgActionDispatchStats::default(),
            // Hybrid per-action flat-view dispatch (ty-side M0, wishlist item 4);
            // inert until lazily initialized and only active under TY_HYBRID_FLAT_VIEW.
            hybrid_dispatch: super::super::hybrid_dispatch::HybridDispatchState::default(),
            // Item 4 M0-G1: hybrid-layout native action cache (TY_HYBRID_NATIVE).
            trust_cg_hybrid_cache: None,
            trust_cg_hybrid_jit_layout: None,
            hybrid_action_out_scratch: Vec::new(),
            checkpoint: CheckpointState {
                dir: None,
                interval: Duration::from_secs(300),
                last_time: None,
                spec_path: None,
                config_path: None,
                spec_hash: None,
                config_hash: None,
            },
            debug: DebugDiagnostics {
                seen_tlc_fp_dedup: if debug_seen_tlc_fp_dedup() {
                    Some(FxHashMap::default())
                } else {
                    None
                },
                seen_tlc_fp_dedup_collisions: 0,
                seen_tlc_fp_dedup_collision_limit: debug_seen_tlc_fp_dedup_collision_limit(),
                #[cfg(debug_assertions)]
                lazy_values_in_state_states: 0,
                #[cfg(debug_assertions)]
                lazy_values_in_state_values: 0,
                #[cfg(debug_assertions)]
                lazy_values_in_state_log_limit: debug_lazy_values_in_state_log_limit(),
                internal_fp_collision: if debug_internal_fp_collision() {
                    Some(FxHashMap::default())
                } else {
                    None
                },
                internal_fp_collisions: 0,
                internal_fp_collision_limit: debug_internal_fp_collision_limit(),
            },

            // POR fields - initialized empty, populated in setup_bfs if por_enabled
            por: super::PorState {
                independence: None,
                visibility: crate::por::VisibilitySet::new(),
                stats: crate::por::PorStats::default(),
                parity_checked_states: 0,
                parity_failed: false,
                actions_populated_for_por: false,
                last_benefit_check_total: 0,
                last_benefit_check_reductions: 0,
            },
            // Part of #2752: periodic liveness checking (TLC doPeriodicWork pattern)
            periodic_liveness: PeriodicLivenessState::default(),
            // Part of #3717: portfolio racing verdict (None = standalone, no portfolio)
            portfolio_verdict: None,
            // Part of #3767: cooperative dual-engine state (None = standalone, no fused mode)
            #[cfg(feature = "ay")]
            cooperative: None,
            // Deferred non-atomic Rp engagement for the fused BFS lane
            // (armed per-loop by `run_bfs_loop`; see mc_struct docs).
            #[cfg(feature = "ay")]
            rp_deferred_nonatomic_armed: false,
            #[cfg(feature = "ay")]
            rp_deferred_poll_tick: 0,
            collision_detector: None,
            // Part of #3986: FlatState layout inferred after init state solving.
            flat_state_layout: None,
            // Part of #3986: FlatBfsBridge created after layout inference.
            flat_bfs_bridge: None,
            // Part of #4126: FlatBfsAdapter created alongside bridge.
            flat_bfs_adapter: None,
            // Part of #3986: Set to true when all vars are scalar and flat state is primary.
            flat_state_primary: false,
            homotopic_canonicalizer: None,
            // WP-11 slice 2: installed post-layout-inference by
            // `maybe_install_flat_symmetry_canonicalizer` (TY_FLAT_SYMMETRY=1
            // gated, fail-closed admission); None keeps the interpreter
            // SymmetryCanonical domain byte-identical.
            flat_symmetry_canonicalizer: None,
            // Nested-set A5: populated by freeze_nested_set_monitors after the
            // discovery prefix; empty for every spec without a set-of-sets var.
            nested_set_monitors: Vec::new(),
            // Step B: native slide kernel; armed at BFS start when the static
            // recognizer proves the spec's Next is the slide relation (the
            // default), or force-armed via `TY_NESTED_SET_SLIDE=1`.
            nested_set_slide_arm: None,
            // First-N-states kernel-vs-interpreter tripwire; set on default
            // (recognizer) arms only.
            nested_set_slide_tripwire: 0,
        }
    }
}
