// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! ParallelChecker constructor methods.

use super::*;

/// Memory-aware ceiling on the number of BFS worker threads (Part of #18).
///
/// Each worker carries its own per-thread value-intern overlays and op_defs
/// working set, so peak memory grows roughly linearly with the worker count.
/// Above ~16 workers the marginal model-checking throughput gain is small while
/// the memory cost keeps climbing, so we cap here to keep the worst case bounded
/// regardless of `available_parallelism()` or a large user `--workers N`.
/// Mirrors the aiger portfolio's `ADAPTIVE_MAX_WORKERS = 16`.
const ADAPTIVE_MAX_WORKERS: usize = 16;

impl ParallelChecker {
    /// Create a new parallel model checker
    ///
    /// # Arguments
    /// * `module` - The TLA+ module to check
    /// * `config` - Model checking configuration
    /// * `num_workers` - Number of worker threads (0 = use number of CPUs)
    pub fn new(module: &Module, config: &Config, num_workers: usize) -> Self {
        Self::new_with_extends(module, &[], config, num_workers)
    }

    /// Create a new parallel model checker with additional loaded modules.
    ///
    /// Despite the historical name, `extended_modules` is **not** "EXTENDS-only".
    /// It must be a *loaded-module superset* for the whole run:
    ///
    /// - Include every non-stdlib module that may be referenced, via `EXTENDS` or `INSTANCE`
    ///   (including transitive and nested instance dependencies).
    /// - Put the modules that contribute to the **unqualified** operator namespace first, in a
    ///   TLC-shaped deterministic order (the `EXTENDS` closure and standalone `INSTANCE` imports).
    ///   Remaining loaded modules may follow in any deterministic order.
    ///
    /// Missing referenced non-stdlib modules are treated as a setup error.
    ///
    /// # Arguments
    /// * `module` - The TLA+ module to check
    /// * `extended_modules` - Modules that the main module extends
    /// * `config` - Model checking configuration
    /// * `num_workers` - Number of worker threads (0 = use number of CPUs)
    pub fn new_with_extends(
        module: &Module,
        extended_modules: &[&Module],
        config: &Config,
        num_workers: usize,
    ) -> Self {
        // Match the sequential direct-constructor boundary. Coordinator-side
        // semantic caches can outlive a prior module even though every worker
        // below is a newly spawned thread with fresh TLS.
        crate::clear_thread_local_eval_caches();
        // Part of #18 (OOM hardening): clamp the worker count to a memory-aware
        // ceiling. Every worker holds its own per-thread value-intern overlays
        // and op_defs working set, so memory scales ~linearly with the worker
        // count. On a high-core machine, an uncapped auto-detect (or a large
        // explicit `--workers N`) multiplies per-worker memory until the process
        // OOMs. We cap at `ADAPTIVE_MAX_WORKERS` (mirroring the aiger portfolio's
        // ceiling) so the worst case is bounded regardless of `available_parallelism`
        // or user input. The cap applies to BOTH the auto-detect (0) path and the
        // explicit path (which previously had no ceiling at all).
        let num_workers = if num_workers == 0 {
            // Part of #3170: With the CAS-based fingerprint set replacing the
            // RwLock-backed Sharded backend, lock contention no longer caps
            // parallel scaling. Use available cores, bounded by the memory ceiling.
            #[allow(clippy::redundant_closure_for_method_calls)]
            thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        } else {
            num_workers
        }
        .clamp(1, ADAPTIVE_MAX_WORKERS);

        // Part of #810: Use shared setup pipeline directly. CheckerSetup already
        // collects vars/op_defs/assumes and resolves state variable references,
        // eliminating the redundant collect + resolve that the previous ModuleSetup
        // wrapper required.
        let CheckerSetup {
            ctx: _ctx,
            main_module,
            rewritten_exts,
            unqualified_modules: unqualified_modules_owned,
            vars,
            op_defs,
            assumes,
            mut setup_error,
        } = setup_checker_modules(
            module,
            extended_modules,
            config,
            &SetupOptions {
                load_instances: false,
            },
        );

        // Create VarRegistry for ArrayState <-> State conversion
        let var_registry = Arc::new(VarRegistry::from_names(vars.iter().cloned()));

        // Part of #4053: Determine if the spec can produce lazy values at runtime.
        let spec_may_produce_lazy =
            crate::materialize::spec_may_produce_lazy_values(module, extended_modules);

        // Part of #1121: Shared conservative Trace detector across checker paths.
        // Parallel mode needs this up-front because `check()` takes `&self`.
        let uses_trace = match compute_uses_trace(config, &op_defs) {
            Ok(val) => val,
            Err(e) => {
                if setup_error.is_none() {
                    setup_error = Some(e);
                }
                false
            }
        };

        // Use 256 shards (2^8) for fingerprint set - reduces lock contention in no-trace mode
        // With 256 shards and 8 workers, collision probability is ~3.1%
        let shard_bits = 8;
        // Part of #2955: Align DashMap shard counts with fingerprint set (256 shards).
        // Default DashMap uses num_cpus*4 shards (~64), causing higher collision
        // probability (~25%) vs 256 shards (~6%) per operation.
        let dashmap_shards = 1 << shard_bits; // 256

        // Part of #3304: propagate env-var parse errors into setup_error
        // instead of panicking.
        let fpset_mode = match parallel_fpset_mode() {
            Ok(mode) => mode,
            Err(e) => {
                if setup_error.is_none() {
                    setup_error = Some(e);
                }
                StorageMode::ShardedCas
            }
        };

        // Part of #3285: Allow overriding FPSet capacity via env var for diagnostic
        // benchmarking of L3 cache effects at different table sizes.
        let fp_capacity = {
            static FP_CAP: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
            *FP_CAP.get_or_init(|| {
                std::env::var("TY_FP_CAPACITY")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
            })
        }
        .or(matches!(&fpset_mode, StorageMode::ShardedCas)
            .then_some(default_parallel_fpset_capacity(num_workers)));

        ParallelChecker {
            num_workers,
            seen: Arc::new(DashMap::with_hasher_and_shard_amount(
                FxBuildHasher::default(),
                dashmap_shards,
            )),
            // Part of #3170/#3285: default to CAS backend unless overridden
            // by TY_PARALLEL_FPSET env var for diagnostic benchmarking.
            seen_fps: FingerprintSetFactory::create(StorageConfig {
                mode: fpset_mode,
                shard_bits,
                capacity: fp_capacity,
                ..Default::default()
            })
            .expect("FPSet storage creation is infallible"),
            // Part of #3178: Per-worker sharded append log replaces DashMap.
            parent_log: Arc::new(ParentLog::new(num_workers)),
            var_registry,
            store_full_states: false,
            stop_flag: Arc::new(AtomicBool::new(false)),
            depth_limit_reached: Arc::new(AtomicBool::new(false)),
            work_remaining: Arc::new(AtomicUsize::new(0)),
            active_workers: Arc::new(AtomicUsize::new(0)),
            max_queue_depth: Arc::new(AtomicUsize::new(0)),
            max_depth: Arc::new(AtomicUsize::new(0)),
            total_transitions: Arc::new(AtomicUsize::new(0)),
            total_raw_initial_states_generated: Arc::new(AtomicUsize::new(0)),
            total_raw_successors_generated: Arc::new(AtomicUsize::new(0)),
            vars,
            op_defs: op_defs.into_iter().collect(),
            config: config.clone(),
            setup_error,
            check_deadlock: config.check_deadlock,
            module: Arc::new(main_module),
            extended_modules: Arc::new(rewritten_exts),
            unqualified_modules: Arc::new(unqualified_modules_owned),
            max_states_limit: None,
            max_depth_limit: None,
            deadline: None,
            progress_callback: None,
            // Part of #3247: 10s matches TLC's default progress interval.
            progress_interval_ms: 10_000,
            continue_on_error: false,
            first_violation: Arc::new(OnceLock::new()),
            first_action_property_violation: Arc::new(OnceLock::new()),
            first_violation_trace: Arc::new(OnceLock::new()),
            states_at_stop: Arc::new(OnceLock::new()),
            admitted_states: Arc::new(AtomicUsize::new(0)),
            collision_diagnostics: ParallelCollisionDiagnostics::from_env(dashmap_shards),
            collision_check_mode: crate::collision_detection::CollisionCheckMode::None,
            assumes,
            uses_trace,
            input_base_dir: None,
            has_run: AtomicBool::new(false),
            run_diagnostics: Arc::new(crate::run_diagnostics::RunDiagnostics::default()),
            successors: if config.has_liveness_properties() {
                Some(Arc::new(DashMap::with_hasher_and_shard_amount(
                    FxBuildHasher::default(),
                    dashmap_shards,
                )))
            } else {
                None
            },
            liveness_init_states: Arc::new(DashMap::with_hasher_and_shard_amount(
                FxBuildHasher::default(),
                dashmap_shards,
            )),
            // Part of #3011: Only allocate when both symmetry AND liveness are active.
            successor_witnesses: if config.has_liveness_properties() && config.symmetry.is_some() {
                Some(Arc::new(DashMap::with_hasher_and_shard_amount(
                    FxBuildHasher::default(),
                    dashmap_shards,
                )))
            } else {
                None
            },
            fairness: Vec::new(),
            stuttering_allowed: true,
            auto_create_trace_file: true,
            file_id_to_path: FxHashMap::default(),
            barrier: Arc::new(WorkBarrier::new(num_workers)),
            depths: Arc::new(DashMap::with_hasher_and_shard_amount(
                FxBuildHasher::default(),
                dashmap_shards,
            )),
            checkpoint_dir: None,
            checkpoint_interval: Duration::from_secs(300),
            periodic_liveness: PeriodicLivenessState::default(),
            checkpoint_spec_path: None,
            checkpoint_config_path: None,
            checkpoint_spec_hash: None,
            checkpoint_config_hash: None,
            promoted_property_invariants: OnceLock::new(),
            memory_policy: None,
            disk_limit_bytes: None,
            internal_memory_limit: None,
            tier_state: OnceLock::new(),
            spec_may_produce_lazy,
            symmetry_disabled_for_liveness: OnceLock::new(),
        }
    }

    /// Test-only accessor for the resolved worker count, so regression tests can
    /// assert the memory-aware clamp (Part of #18) without exposing the field.
    #[cfg(test)]
    pub(crate) fn worker_count_for_test(&self) -> usize {
        self.num_workers
    }
}

#[cfg(test)]
mod worker_clamp_tests {
    use super::*;
    use crate::test_support::parse_module;

    const TRIVIAL_SPEC: &str = r#"
---- MODULE WorkerClampTrivial ----
VARIABLE x
Init == x = 0
Next == x' = x
====
"#;

    /// Regression for bug #18 (OOM hardening): an explicit `--workers N` above
    /// the memory-aware ceiling must be clamped, not honored. Before the fix the
    /// explicit path had no ceiling, so a user could spawn one heavy worker per
    /// requested thread (each with its own intern overlays + op_defs), blowing
    /// up memory. We pass an absurd count and assert it is capped at
    /// `ADAPTIVE_MAX_WORKERS` — validated WITHOUT actually spawning thousands of
    /// threads (the clamp happens at construction, before any worker starts).
    #[test]
    fn explicit_worker_count_is_clamped_to_ceiling() {
        let module = parse_module(TRIVIAL_SPEC);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };

        let checker = ParallelChecker::new(&module, &config, 100_000);
        assert_eq!(
            checker.worker_count_for_test(),
            ADAPTIVE_MAX_WORKERS,
            "explicit oversized --workers must be clamped to the memory ceiling"
        );
    }

    /// The auto-detect path (`num_workers == 0`) must also respect the ceiling:
    /// `available_parallelism()` on a high-core box could otherwise spawn far
    /// more workers than the memory budget supports.
    #[test]
    fn auto_worker_count_respects_ceiling() {
        let module = parse_module(TRIVIAL_SPEC);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };

        let checker = ParallelChecker::new(&module, &config, 0);
        let n = checker.worker_count_for_test();
        assert!(
            (1..=ADAPTIVE_MAX_WORKERS).contains(&n),
            "auto-detected worker count {n} must be within 1..={ADAPTIVE_MAX_WORKERS}"
        );
    }

    /// A modest explicit count below the ceiling must pass through unchanged so
    /// the clamp does not regress normal usage.
    #[test]
    fn small_explicit_worker_count_is_unchanged() {
        let module = parse_module(TRIVIAL_SPEC);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };

        let checker = ParallelChecker::new(&module, &config, 4);
        assert_eq!(checker.worker_count_for_test(), 4);
    }
}
