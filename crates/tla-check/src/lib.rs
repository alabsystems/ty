// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! tla-check - TLA+ Model Checker
//!
//! This crate provides:
//! - **Value types**: Re-exported from `tla-value` crate (`value/`)
//! - **Expression evaluator**: Re-exported from `tla-eval` crate (`eval/`)
//! - **State exploration**: BFS state space exploration with parallel workers (`parallel/`)
//! - **Safety checking**: Invariant verification, action constraints, state constraints
//! - **Liveness checking**: Tableau construction, SCC detection, WF/SF fairness (`liveness/`)
//! - **Trace validation**: Verify system traces against specs
//! - **Configuration parsing**: Parse TLC .cfg files (`config/`)
//! - **Checkpoint/resume**: Save and restore model checking state
//! - **Mmap storage**: Memory-mapped state storage for large state spaces (`storage/`)
//! - **POR prototype**: Static dependency and visibility analysis scaffolding (`por/`);
//!   transition reduction is currently disabled pending a sound cycle proviso
//!
//! # Quick Start
//!
//! ```rust
//! use tla_check::{eval, EvalCtx, Value};
//! use tla_core::{lower, parse_to_syntax_tree, FileId};
//!
//! // Parse and evaluate a TLA+ expression
//! let src = "---- MODULE Test ----\nOp == 1 + 2 * 3\n====";
//! let tree = parse_to_syntax_tree(src);
//! let result = lower(FileId(0), &tree);
//! let module = result.module.unwrap();
//!
//! // Find the operator and evaluate it
//! let mut ctx = EvalCtx::new();
//! ctx.load_module(&module);
//!
//! // The result of 1 + 2 * 3 is 7
//! ```
//!
//! # Model Checking
//!
//! ```rust,no_run
//! use tla_check::{check_module, CheckResult, Config};
//! use tla_core::{lower, parse_to_syntax_tree, FileId};
//!
//! // A tiny, finite spec (2 reachable states)
//! let src = "---- MODULE Counter ----\n\
//! VARIABLE x\n\
//! Init == x = 0\n\
//! Next == x' = x + 1 /\\ x < 1\n\
//! ====";
//! let tree = parse_to_syntax_tree(src);
//! let lowered = lower(FileId(0), &tree);
//! let module = lowered.module.unwrap();
//!
//! let config = Config::parse("INIT Init\nNEXT Next\n").unwrap();
//! let result = check_module(&module, &config);
//! match result {
//!     CheckResult::Success(stats) => println!("OK: {} states", stats.states_found),
//!     CheckResult::InvariantViolation { invariant, trace, .. } => {
//!         println!("Violated: {}\n{}", invariant, trace);
//!     }
//!     other => println!("Result: {:?}", other),
//! }
//! ```

#![deny(missing_docs)]
#![allow(clippy::result_large_err)]

// debug_flag! macro must be defined before modules that use it
#[macro_use]
pub(crate) mod debug_env;
pub use debug_env::{set_telemetry_quiet, telemetry_quiet};

pub(crate) mod action_instance;
pub(crate) mod adaptive;
pub mod analytical;

/// Single blessed choke point for process-environment mutation (test/CLI
/// plumbing). Always compiled so both in-crate `#[cfg(test)]` tests and
/// out-of-crate integration tests reach the same choke point.
#[doc(hidden)]
pub mod env_guard;
pub(crate) mod arena;
mod cfg_overrides;
pub(crate) mod check;
pub(crate) mod checker_ops;
pub(crate) mod checker_setup;
pub(crate) mod checkpoint;
pub mod collision_detection;
pub(crate) mod compiled_backend_unavailable;
pub(crate) mod complexity_visitor;
pub(crate) mod config;
pub(crate) mod constants;
pub(crate) mod coverage;
mod disabled_action_stats;
pub(crate) mod enabled;
pub(crate) mod enumerate;
pub(crate) mod error;
pub(crate) mod error_policy;
/// Compatibility shim — prefer crate-root exports or direct `tla_eval` imports (#3039).
#[doc(hidden)]
pub mod eval;
pub(crate) mod expr_visitor;
pub(crate) mod fingerprint;
mod guard_error_stats;
pub(crate) mod init_strategy;
pub(crate) mod intern;
pub mod itf;
pub(crate) mod json_codec;
pub(crate) mod json_output;
mod json_path;
pub(crate) mod liveness;
pub(crate) mod materialize;
pub(crate) mod memory;
pub(crate) mod parallel;
pub(crate) mod periodic_liveness;
pub(crate) mod run_diagnostics;
// Distributed model checking — state partitioning prototype (Part of #3796).
// Off-by-default experimental subsystem with no production entry point (only its own
// `#[cfg(test)]` tests construct it), so it is excluded from release builds and only
// compiled under `cargo test` or the `distributed-experimental` feature.
#[cfg(any(test, feature = "distributed-experimental"))]
#[allow(dead_code)]
pub(crate) mod distributed;
pub mod resource_budget;
#[cfg(test)]
mod resource_budget_tests;
pub(crate) mod spec_formula;
pub mod state;
pub(crate) mod storage;
pub(crate) mod tir_mode;
pub(crate) mod trace_action_label_mapping;
pub(crate) mod trace_action_label_rewrite;
pub mod trace_explain;
pub(crate) mod trace_file;
pub(crate) mod trace_input;
pub(crate) mod trace_mine;
pub(crate) mod trace_validate;
/// Compatibility shim — prefer crate-root exports or direct `tla_value` imports (#3039).
#[doc(hidden)]
pub mod value;
pub(crate) mod var_index;
// Vacuity-gate verdict/warning types (design: TRUST_VACUITY_GATE §1.A).
pub mod vacuity;
// State space estimation from BFS level statistics
pub mod state_space_estimator;
// Spec minimizer: delta debugging for TLA+ specs
pub mod minimize;
// Incremental model checking: reuse prior results when spec changes
pub mod incremental_check;
// Certifying verification: serialized, independently re-checkable inductive-safety certificates
pub mod cert;
// Zero-arity operator / configured-CONSTANT inlining for the certificate recognizers (the
// deterministic front door that lets `can \in Can`-shaped bodies reach the kernel fragment).
pub(crate) mod cert_inline;
// `ty certify` decline EXPLANATION: re-run the certification pipeline's stages one by one and
// report every stage that fails, with specifics (north star: "wherever ty cannot deliver, it
// says so exactly").
pub mod certify_explain;
// ty.verdict/v1: a content-addressed, independently re-checkable verdict envelope
// (VIOLATED direction — replayable counterexample trace, eval-only re-check).
pub mod verdict;
// Deadlock analysis engine
pub mod deadlock_analysis;
// Self-liveness: TY's engine-selection control flow as a structured temporal
// obligation (DATA), rendered to TLA+ and discharged by TY's own check_module.
// Roadmap step 1 of docs/design/trust-verification-atoms-2026-06-17.md.
pub mod selfliveness;
// Portfolio racing verdict for parallel BFS + ay PDR/BMC (Part of #3717)
pub mod shared_verdict;
// Kernel-checkable `Certified` tier: build a CIC proof term + type-check it with the small Clean
// kernel (fail-closed). Opt-in (`clean-cic` feature) — shrinks a verdict's trust base to the kernel.
#[cfg(feature = "clean-cic")]
pub(crate) mod cleancic;
// Phase 1 (docs/kernel-checked-tla-plan.md): clean-ck0 — the genuinely tiny, independently
// auditable kernel — as the MANDATORY second checker for the Nat/Bool obligation fragment.
// A ck0 rejection of a clean-kernel-accepted term fails certification closed; the tiny-TCB
// label attaches only to ck0-corroborated legs.
#[cfg(feature = "clean-cic")]
pub mod ck0_bridge;
// Plan item B#3 (docs/kernel-checked-tla-plan.md): the honest trust census of the REAL kernel
// env `Certified` verdicts run over — surfaces the prelude's trust markers + domain axioms that
// the curated "3-axiom" certificate env does not represent.
#[cfg(feature = "clean-cic")]
pub mod kernel_census;
// ADEQUACY ROAD v1 (docs/kernel-checked-tla-plan.md, Phase C increment): the DEEP-EMBEDDING
// corroboration lane — PredIR quoted constructor-for-constructor into kernel-admitted syntax
// inductives and decided by a kernel-checked evaluator, shrinking the trusted embedder to a
// line-auditable quoter for reflected obligations. STANDALONE lane today (tested, not yet
// wired into the certify paths — that wiring, and the fuel-fold set/quantifier fragment,
// is the v2 step).
#[cfg(feature = "clean-cic")]
pub mod reflect;
// The reflected `R ⊆ Safety` re-check leg: an ADDITIVE, embedder-free re-discharge of an
// explicit-state fixpoint cert's safety obligation through the kernel-defined `reflect` evaluator
// (the `ty reflect-check` surface). Verdict type is always compiled; the re-check entry point is
// `clean-cic`-gated inside (needs the linked kernel + the spec re-derivation).
pub mod reflect_safety_check;
// AST-DIRECT reflected discharge (design pivot increment 1, docs/cert/design-pivot-reflect.md):
// quote Init/Next/Safety DIRECTLY from the (re-parsed, operator-inlined) TLA+ AST into the
// kernel's deep embedding — retiring the RECOGNIZER from the reflect-check trust base for the
// covered fragment (the `ty reflect-check --ast-direct` surface). `clean-cic`-gated (needs the
// linked kernel).
#[cfg(feature = "clean-cic")]
pub mod reflect_ast_direct;
// KERNEL-CERTIFIED REFINEMENT MAPPINGS (`Spec_impl ⇒ Spec_abs`, safety part): the abstract
// Init/Next are kernel-re-evaluated at every mapped implementation init/transition (with the
// stuttering disjunct `[Next_abs]_vars` requires). Always compiled (serde schema); certify/
// verify entry points are `clean-cic`-gated inside.
pub mod refinement_cert;
// LIVE explicit-state fixpoint Certified verdict: hook the model checker's ENUMERATED reachable set
// into the kernel fixpoint cert. Always compiled (so the default build's module graph is unchanged);
// its certifying entry points are `clean-cic`-gated inside, fail-closed otherwise.
pub mod explicit_fixpoint_cert;
// Partial Order Reduction (Part of #541)
pub(crate) mod por;
// ay-based Init enumeration (Part of #251)
#[cfg(feature = "ay")]
pub(crate) mod ay_enumerate;
// ay-based nested powerset Init enumeration (Part of #3826)
#[cfg(feature = "ay")]
pub(crate) mod ay_init_powerset;
// Shared sort inference for ay-based symbolic checking
#[cfg(feature = "ay")]
pub(crate) mod ay_shared;
// ay-based PDR symbolic safety checking (Part of #642)
#[cfg(feature = "ay")]
pub(crate) mod ay_pdr;
// ay-based bounded model checking (Part of #3702)
#[cfg(feature = "ay")]
pub(crate) mod ay_bmc;
// Second, independent TLA front end for the scalar certifiable fragment — the
// third (front-end-independent) Leg D part-2 binding for certifying verification.
#[cfg(feature = "ay")]
pub(crate) mod cert_indep_frontend;
// Re-checkable LIVENESS certificates (well-founded descent for <>P under WF).
//   * the AY lane (5 SMT obligations + descent kernel leg) needs `ay`;
//   * the EXPLICIT-STATE KERNEL lane (`certify_liveness_explicit`: descent + bounded-below +
//     enabledness-at-terminals kernel-checked over the enumerated reachable set, NO solver) needs
//     only `clean-cic`. Compile the module whenever EITHER lane is enabled; each lane's items are
//     feature-gated internally.
#[cfg(any(feature = "ay", feature = "clean-cic"))]
pub mod live_cert;
// ALL-N (parametric) certificates: prove an invariant for every value of a
// scalar CONSTANT by promoting it to a rigid free variable.
pub mod cert_all_n;
// ay-based k-induction for proving safety properties (Part of #3722)
#[cfg(feature = "ay")]
pub(crate) mod ay_kinduction;
// ay-based inductive invariant convenience checker (Apalache Gap 8, Part of #3756)
#[cfg(feature = "ay")]
pub(crate) mod ay_inductive_check;
// VMT (Verification Modulo Theories) export for external model checkers (Part of #3755)
#[cfg(feature = "ay")]
#[allow(dead_code)]
pub(crate) mod vmt_export;
// Cooperative Dual-Engine Model Checking shared state (Part of #3762)
// Extracted sub-modules for cooperative_state (Part of #3872)
#[cfg(feature = "ay")]
pub(crate) mod action_metrics;
#[cfg(feature = "ay")]
pub(crate) mod convergence;
#[cfg(feature = "ay")]
pub(crate) mod cooperative_state;
#[cfg(feature = "ay")]
pub(crate) mod per_invariant;
#[cfg(feature = "ay")]
pub(crate) mod smt_compat;
// ay-based symbolic simulation (Apalache Gap 9, Part of #3757)
#[cfg(feature = "ay")]
pub(crate) mod ay_symbolic_sim;

// Interactive JSON-RPC server for step-by-step state exploration (Apalache Gap 3, Part of #3751)
pub mod interactive;
// Symbolic exploration engine for interactive SMT-based exploration (Apalache Gap 3, Part of #3751)
#[cfg(feature = "ay")]
pub(crate) mod symbolic_explore;

// Kani formal verification harnesses (only compiled with cargo kani)
#[cfg(kani)]
mod kani_harnesses;

// Regular tests that mirror Kani proofs (excluded during kani runs to avoid duplicate)
#[cfg(all(test, not(kani)))]
mod kani_harnesses;

/// Process-wide lock serializing test access to process-global state
/// (environment variables in particular: `std::env::set_var`/`remove_var`
/// race against every other thread's `var` under parallel `cargo test`).
/// Poisoning is deliberately ignored — a panicking test must not cascade
/// lock failures into unrelated tests.
#[cfg(test)]
#[allow(dead_code)] // consumed by env-mutating test modules as they adopt it
pub(crate) fn process_env_lock() -> std::sync::MutexGuard<'static, ()> {
    env_guard::lock_env()
}

// Re-exports
pub use adaptive::{check_module_adaptive, AdaptiveChecker, PilotAnalysis, Strategy};
#[cfg(feature = "testing")]
pub use check::StateGraphSnapshot;
pub use check::{
    check_module, resolve_spec_from_config, resolve_spec_from_config_with_extends, ActionLabel,
    CheckError, CheckResult, CheckStats, ConfigCheckError, EvalCheckError, InfraCheckError,
    LimitType, LivenessCheckError, ModelChecker, PorReductionStats, Progress, ProgressCallback,
    PropertyCheckStats, PropertyViolationKind, RandomWalkConfig, RandomWalkResult, ResolvedSpec,
    RuntimeCheckError, SimulationConfig, SimulationResult, SimulationStats, SuccessorWitnessMap,
    SymmetryReductionStats, Trace,
};
// GPU engine admission (safety-only, fail-closed): the plain-typed program
// handed to the `tla-gpu` CUDA engine by `ty check --gpu`.
pub use check::model_checker::gpu_admit::{GpuFunction, GpuProgram};
// Multi-phase verification pipeline (Part of #3723)
pub use check::pipeline::{
    PhaseConfig, PhaseRecord, PhaseRunner, PipelineResult, PropertyVerdict, VerificationPhase,
    VerificationPipeline, VerificationStrategy,
};
// Concrete pipeline runners (Part of #3720, #3723)
pub use check::pipeline_runners::{BfsRunner, RandomWalkRunner};
#[cfg(feature = "ay")]
pub use check::pipeline_runners::{BmcRunner, KInductionRunner, PdrRunner};
// Portfolio orchestrator for parallel BFS + PDR (Part of #3717)
pub use check::portfolio::{run_portfolio, PortfolioResult, PortfolioWinner};
pub use config::{
    Config, ConfigError, ConstantValue, InitMode, LivenessExecutionMode, TerminalSpec,
};
pub use constants::{bind_constants_from_config, parse_constant_value};
pub use error::{EvalError, EvalResult};
pub use eval::{aggregate_bytecode_vm_stats, bytecode_vm_stats};
pub use eval::{eval, Env, EvalCtx, OpEnv};
#[cfg(any(test, feature = "testing"))]
pub use liveness::debug::{
    set_liveness_disk_bitmask_flush_threshold_override, set_liveness_inmemory_node_limit_override,
    set_liveness_inmemory_successor_limit_override, set_use_disk_bitmasks_override,
    set_use_disk_graph_override, set_use_disk_successors_override,
    LivenessDiskBitmaskFlushThresholdGuard, LivenessInMemoryNodeLimitGuard,
    LivenessInMemorySuccessorLimitGuard, UseDiskBitmasksGuard, UseDiskGraphGuard,
    UseDiskSuccessorsGuard,
};
pub use parallel::{check_module_parallel, ParallelChecker};
#[cfg(any(test, feature = "testing"))]
pub use parallel::{set_use_handle_state_override, UseHandleStateGuard};
pub use resource_budget::{ResourceBudget, StateSpaceEstimate};
pub use spec_formula::{extract_spec_formula, FairnessConstraint, SpecFormula};
pub use state::{
    fp_hashmap, fp_hashmap_with_capacity, fp_hashset, fp_hashset_with_capacity, value_fingerprint,
    ArrayState, BuildFingerprintHasher, Fingerprint, FingerprintHasher, FpHashMap, FpHashSet,
    State,
};
pub use value::{FuncSetValue, FuncValue, IntervalValue, SubsetValue, Value};
// Explicit root export for memory_stats so CLI callers don't need the shim (#3039).
#[cfg(feature = "memory-stats")]
pub use value::memory_stats;

// Liveness checking — integration tests use AstToLive + ConvertError
pub use liveness::{AstToLive, ConvertError};

// Checker operations (safety/temporal property classification)
pub use checker_ops::any_property_requires_liveness_checker;
// TLC-parity action-label attribution for replay-validated symbolic
// counterexample traces (fused verdict-race promotion; #2470/#2920 parity).
pub use checker_ops::label_trace_actions;

// Coverage statistics
pub use coverage::{detect_actions, ActionStats, CoverageStats, DetectedAction, DetectedActionId};
pub use vacuity::{VacuityClass, VacuityReason, VacuityWarning};

// TLC-style action instances (trace validation / diagnostics)
pub use action_instance::{
    enumerate_successors_by_action_instance, split_action_instances, ActionInstance,
    ActionInstanceSuccessors,
};

// Init enumeration strategy selection (Part of #251)
pub use init_strategy::{analyze_init_predicate, AYReason, InitAnalysis, InitStrategy};

// Collision detection for fingerprint-based state storage
pub use collision_detection::{CollisionCheckMode, CollisionDetail, CollisionStats};

// Trace explanation engine
pub use trace_explain::{ExplainedStep, ExplainedTrace, TraceStateDiff, VarChange};

// Scalable storage
pub use storage::factory::{FingerprintSetFactory, StorageConfig, StorageMode};
pub use storage::{
    CapacityStatus, DiskFingerprintSet, FingerprintSet, FingerprintStorage, InsertOutcome,
    LookupOutcome, ShardedFingerprintSet, StorageFault, StorageStats, TraceLocationStorage,
    TraceLocationsStorage,
};

// Disk-based trace storage
pub use trace_file::{TraceFile, TraceLocations};

// Trace validation mapping config (action labels)
pub use trace_action_label_mapping::{
    validate_action_label, ActionLabelInstanceError, ActionLabelMapping, ActionLabelMappingConfig,
    ActionLabelValidationError, ActionLabelValidationResult, ActionParamEncoding, OperatorRef,
    TraceActionLabelMappingError,
};
pub use trace_action_label_rewrite::{
    outer_exists_binder_sites, resolve_param_binders_for_rewrite, ActionLabelParamBindError,
    ActionLabelParamBindErrorKind, BoundVarSite,
};

// Trace validation input parsing
pub use trace_input::{
    read_trace_events, read_trace_json, read_trace_jsonl, resolve_trace_input_format,
    TraceActionLabel, TraceEventSink, TraceHeader, TraceInputFormat, TraceInputFormatSelection,
    TraceParseError, TraceSourceHint, TraceStep,
};

// Spec mining v1 (`ty trace mine`): candidate specs from trace corpora
pub use trace_mine::{
    mine_spec, render_config, render_module, ActionUpdate, ClusterKind, InvariantKind, MineError,
    MineOptions, MinedAction, MinedInvariant, MinedProperty, MinedSpec, MiningTrace, UpdatePattern,
    VarDomain,
};

// Variable indexing
pub use var_index::{VarIndex, VarRegistry};

// Global state reset and memory/disk policy helpers (Part of #3608)
mod reset;
pub use reset::{
    clear_thread_local_eval_caches, disk_limit_system_default, log_memory_budget,
    memory_policy_system_default, memory_policy_system_default_info, reset_global_state,
};

/// Process-wide lifecycle guard for an independently parsed model-check input.
///
/// Acquire this *before* parsing or lowering and retain it through checking and
/// replay. It prevents [`reset_global_state`] from invalidating global NameIds
/// or ValueHandles during that whole window. `check_module` acquires its own
/// nested run guard for execution, but that is intentionally too late to cover
/// caller-owned parsing.
#[must_use = "retain the context guard through parse, lower, check, and replay"]
pub struct ModelCheckContextGuard {
    _run: intern::ModelCheckRunGuard,
}

/// Enter a reset-safe model-check context before constructing semantic input.
pub fn enter_model_check_context() -> ModelCheckContextGuard {
    ModelCheckContextGuard {
        _run: intern::ModelCheckRunGuard::begin(),
    }
}

// Checkpoint/resume support
pub use checkpoint::{
    Checkpoint, CheckpointMetadata, CheckpointStats, SerializableState, SerializableValue,
};

// JSON output format for AI agents
pub use json_codec::{
    json_to_value, json_to_value_with_path, params_from_json, JsonValueDecodeError,
    JsonValueDecodeErrorAtPath, ParamsDecodeError,
};
pub use json_output::error_codes;
pub use json_output::{
    liveness_trace_to_dot, trace_to_dot, value_to_json, ActionInfo, ActionRef, CounterexampleInfo,
    DiagnosticMessage, DiagnosticsInfo, ErrorSuggestion, InputInfo, JsonOutput, JsonValue,
    JsonlEvent, PrintOutput, PropertyCheckStatsInfo, ResultInfo, SearchCompleteness, SoundnessMode,
    SoundnessProvenance, SourceLocation, SpecInfo, StateDiff, StateInfo, StatisticsInfo,
    StorageStatsInfo, ValueChange, ViolatedProperty, OUTPUT_VERSION,
};

// ITF (Informal Trace Format) trace serialization (Part of #3781, #3753)
pub use itf::{liveness_trace_to_itf, trace_to_itf};

// Trace validation engine (spec-based validation)
pub use trace_validate::{
    ActionLabelMode, ActionMatchResult, StepDiagnostic, TraceValidationBuilder,
    TraceValidationEngine, TraceValidationError, TraceValidationResult, TraceValidationSuccess,
    TraceValidationWarning,
};

// PDR-based symbolic safety checking (Part of #642)
#[cfg(feature = "ay")]
pub use ay_pdr::{
    check_pdr, check_pdr_with_config, check_pdr_with_config_and_evidence, check_pdr_with_evidence,
    check_pdr_with_portfolio, check_pdr_with_portfolio_and_evidence, expand_operators_for_chc,
    PdrError, PdrResult, PdrRunResult,
};
/// Register the full EXTENDS-closure state-variable set into `ctx` for symbolic
/// checking. Callers that hand a top (MC-wrapper) module to the symbolic engines
/// (`check_pdr` / `check_bmc` / `check_kinduction`) must call this after loading
/// `module` and its `checker_modules`, so the real `VARIABLES` declared in an
/// EXTENDS-inherited base module are visible (otherwise the engines bail with
/// "No state variables declared" and the spec silently falls back to BFS).
#[cfg(feature = "ay")]
pub fn register_state_vars_for_symbolic(
    ctx: &mut EvalCtx,
    module: &tla_core::ast::Module,
    checker_modules: &[&tla_core::ast::Module],
) {
    ay_shared::register_state_vars_from_modules(ctx, module, checker_modules);
}

// BMC-based symbolic bug finding (Part of #3702)
#[cfg(feature = "ay")]
pub use ay_bmc::{check_bmc, check_bmc_with_portfolio, BmcConfig, BmcError, BmcResult};
#[cfg(feature = "ay")]
pub use tla_ay::{BmcState, BmcValue};
// k-Induction-based symbolic safety proving (Part of #3722)
#[cfg(feature = "ay")]
pub use ay_kinduction::{
    check_kinduction, check_kinduction_with_portfolio, kinduction_shared_engine_lane_evidence,
    KInductionConfig, KInductionError, KInductionResult,
};
// Inductive invariant convenience checker (Apalache Gap 8, Part of #3756)
#[cfg(feature = "ay")]
pub use ay_inductive_check::{
    check_inductive, InductiveCheckConfig, InductiveCheckError, InductiveCheckResult,
    InductivePhase,
};
// Fused cooperative BFS + symbolic orchestrator (Part of #3769, Epic #3762)
// Available with or without ay: non-ay path runs BFS-only.
pub use check::fused::{
    run_fused_check, run_fused_check_with_config, FusedCheckerConfig, FusedResult, FusedWinner,
    ReconciledVerdict, SymbolicDegradation,
};
// Static complexity analysis for exponential state space detection (Part of #3826)
#[cfg(feature = "ay")]
pub use check::oracle::{detect_exponential_complexity, ExponentialComplexity};
// Counterexample cross-validation for fused orchestrator (Part of #3836)
#[cfg(feature = "ay")]
pub use check::cross_validation::{
    cross_validate_bmc_trace, CrossValidationResult, CrossValidationSource,
};
// VMT export for external model checkers (Apalache Gap 7, Part of #3755)
#[cfg(feature = "ay")]
pub use vmt_export::{export_vmt, VmtError, VmtOutput};
// Symbolic simulation (Apalache Gap 9, Part of #3757)
#[cfg(feature = "ay")]
pub use ay_symbolic_sim::{
    symbolic_simulate, SymbolicSimConfig, SymbolicSimError, SymbolicSimResult, SymbolicSimRunner,
};
// Interactive JSON-RPC server (Apalache Gap 3, Part of #3751)
pub use interactive::{
    InteractiveServer, InteractiveServerError, ServerExploreMode, StateSnapshot,
    DEFAULT_MAX_SYMBOLIC_DEPTH, DEFAULT_PORT,
};

// State space estimation from BFS level statistics
pub use state_space_estimator::{GrowthModel, StateSpaceEstimateResult, StateSpaceEstimator};

// Spec minimizer
pub use minimize::{MinimizeConfig, MinimizeResult, MinimizeStats, Minimizer, ResultClass};

// Incremental model checking (reuse prior results when spec changes)
pub use incremental_check::{
    compute_action_hashes, identify_dirty_states, save_incremental_cache, try_load_incremental,
    ActionProvenance, IncrementalCache, IncrementalCacheMetadata, SpecDiff, SuccessorMap,
};

// Deadlock analysis engine
pub use deadlock_analysis::{analyze_deadlock, ActionDiagnostic, ConjunctResult, DeadlockAnalysis};

/// Shared parse/lower helpers for unit tests within tla-check.
/// Part of #2816: canonical helpers replacing 20+ per-file duplicates.
#[cfg(test)]
pub(crate) mod test_support;

/// Test-only utilities shared across all test modules in tla-check.
/// Part of #3300: Provides the interner serialization lock.
#[cfg(test)]
pub(crate) mod test_utils {
    /// Process-global lock to serialize tests that exercise the parallel checker pipeline.
    /// Required because freeze/unfreeze of the value interner uses process-global state
    /// (PARALLEL_INTERN_ACTIVE, FROZEN_SNAPSHOT) that cannot be isolated between concurrent
    /// test threads. Without this, concurrent `check()` calls cause FROZEN_SNAPSHOT mutex
    /// poisoning when one test's unfreeze races with another's freeze/install cycle.
    pub static PARALLEL_CHECK_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Narrow lock for tests that require process-wide model-check quiescence.
    ///
    /// Keep this separate from `PARALLEL_CHECK_LOCK`: some sequential tests run
    /// long enough that cleanup/reset tests must not start their quiescence wait
    /// while those checks are active, but they should not block unrelated
    /// parallel-checker tests while merely waiting to enter the long section.
    pub static MODEL_CHECK_QUIESCENCE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Acquire the interner serialization lock. Returns a MutexGuard that releases on drop.
    /// Uses `unwrap_or_else(into_inner)` to recover from a poisoned mutex.
    pub fn acquire_interner_lock() -> std::sync::MutexGuard<'static, ()> {
        PARALLEL_CHECK_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Acquire the model-check quiescence lock.
    pub fn acquire_model_check_quiescence_lock() -> std::sync::MutexGuard<'static, ()> {
        MODEL_CHECK_QUIESCENCE_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }
}
