// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::{
    Arc, ArrayState, CapacityStatus, CheckError, CheckStats, Config, DetectedAction, Duration,
    EvalCtx, Expr, FairnessConstraint, Fingerprint, FingerprintSet, FpHashMap, FxHashMap,
    InitProgressCallback, InlineLivenessPropertyPlan, Instant, LiveExpr, LivenessMode, OperatorDef,
    PathBuf, ProgressCallback, Spanned, TraceFile, TraceLocationStorage, TraceLocationsStorage,
};
use crate::storage::{
    ActionBitmaskMap, FingerprintPayloadWitnesses, StateBitmaskMap, SuccessorGraph,
};
// Part of #4398: import fail-closed compatibility types from tla-check's
// local backend shim so source fields no longer depend on deleted packages.
use crate::compiled_backend_unavailable::{
    JitInvariantCache as JitInvariantCacheImpl, JitNextStateCache as JitNextStateCacheImpl,
    RecompilationController as RecompilationControllerImpl, TierManager as TierManagerImpl,
};
// Part of #4171 / #4267: the BFS level loop runs against backend-agnostic
// traits so it remains independent of retired backend crates and can plug in
// trust-codegen without rewriting the loop.
use super::bfs::compiled_step_trait::{
    CompiledBfsLevel as CompiledBfsLevelTrait, CompiledBfsStep as CompiledBfsStepTrait,
};

pub(super) use super::liveness::inline_fairness::EnabledActionGroup;
pub(super) use super::liveness::inline_fairness::EnabledProvenanceEntry;
pub(super) use super::liveness::inline_fairness::SubscriptActionPair;

/// Part of #4128: Identity hash for Fingerprint-keyed maps (pre-hashed keys).
pub(super) type SuccessorWitnessCache = FpHashMap<Vec<(Fingerprint, ArrayState)>>;

/// Checkpoint state for periodic model checking saves.
pub(super) struct CheckpointState {
    /// Directory for saving checkpoints during model checking
    pub(super) dir: Option<PathBuf>,
    /// Interval between checkpoints (in seconds)
    pub(super) interval: Duration,
    /// Time of last checkpoint
    pub(super) last_time: Option<Instant>,
    /// Spec path for checkpoint metadata
    pub(super) spec_path: Option<String>,
    /// Config path for checkpoint metadata
    pub(super) config_path: Option<String>,
    /// SHA-256 hash of spec file content (for checkpoint resume validation)
    pub(super) spec_hash: Option<String>,
    /// SHA-256 hash of config file content (for checkpoint resume validation)
    pub(super) config_hash: Option<String>,
}

/// Module metadata: names, paths, assumptions, variable names, and operator definitions.
pub(super) struct ModuleState {
    /// Root module name (for TLC-style location rendering fallbacks).
    pub(super) root_name: String,
    /// Mapping from FileId to module name for TLC-style location rendering.
    pub(super) file_id_to_name: FxHashMap<tla_core::FileId, String>,
    /// Mapping from FileId to source path for TLC-style line/col location rendering.
    ///
    /// When absent for a given FileId (or unreadable), error locations fall back to byte offsets.
    pub(super) file_id_to_path: FxHashMap<tla_core::FileId, std::path::PathBuf>,
    /// Whether the root input came from Quint JSON IR lowered into the TLA AST.
    pub(super) frontend_source_is_quint: bool,
    /// Setup/configuration error detected during construction.
    ///
    /// Used for config validation that requires access to the loaded module set,
    /// such as TLC-style module-scoped overrides.
    pub(super) setup_error: Option<CheckError>,
    /// ASSUME statements collected from all modules (main + extended)
    /// Each entry is (module_name, assume_expr) for error reporting
    pub(super) assumes: Vec<(String, Spanned<Expr>)>,
    /// Variable names
    pub(super) vars: Vec<Arc<str>>,
    /// Cached operator definitions
    pub(super) op_defs: FxHashMap<String, OperatorDef>,
}

impl<'a> ModelChecker<'a> {
    /// Preserve source-family provenance for descriptor/evidence emission after
    /// a frontend has lowered into the TLA AST consumed by this checker.
    pub fn set_frontend_source_is_quint(&mut self, is_quint: bool) {
        self.module.frontend_source_is_quint = is_quint;
    }
}

/// Partial Order Reduction (POR) prototype state.
pub(super) struct PorState {
    /// Pre-computed independence matrix for POR analysis.
    pub(super) independence: Option<crate::por::IndependenceMatrix>,
    /// Visibility set extracted from invariant expressions.
    pub(super) visibility: crate::por::VisibilitySet,
    /// POR prototype statistics.
    pub(super) stats: crate::por::PorStats,
    /// Fail-closed parity self-check: number of states for which the
    /// per-action successor union has been verified byte-equal against
    /// whole-Next enumeration. Checked for the first
    /// [`POR_PARITY_CHECK_STATES`](super::run_gen::POR_PARITY_CHECK_STATES)
    /// states after POR engagement.
    pub(super) parity_checked_states: u32,
    /// Set when the parity self-check found a mismatch or the per-action
    /// path errored. Permanently disables POR and per-action successor
    /// dispatch for this run (falls back to whole-Next enumeration).
    pub(super) parity_failed: bool,
    /// Whether `setup_actions_and_por` populated `coverage.actions` solely
    /// for POR's per-action enumeration (no coverage/JIT consumer). Allows
    /// the low-benefit auto-POR release to retire them and return to the
    /// faster whole-Next path.
    pub(super) actions_populated_for_por: bool,
    /// Sliding-window snapshot for the auto-POR benefit check:
    /// `stats.total_states` at the last window boundary.
    pub(super) last_benefit_check_total: u64,
    /// `stats.reductions` at the last window boundary.
    pub(super) last_benefit_check_reductions: u64,
}

/// Symmetry reduction state.
pub(super) struct SymmetryState {
    /// Symmetry permutations for state reduction (empty if no SYMMETRY in config)
    pub(super) perms: Vec<crate::value::FuncValue>,
    /// Fast symmetry permutations using MVPerm for O(1) model value lookup (Part of #358)
    pub(super) mvperms: Vec<crate::value::MVPerm>,
    /// Cache: original fingerprint -> canonical fingerprint (for symmetry reduction)
    /// Avoids recomputing canonical fingerprint for the same state
    pub(super) fp_cache: FxHashMap<Fingerprint, Fingerprint>,
    /// Number of fp_cache hits (state fingerprint already known canonical).
    pub(super) fp_cache_hits: u64,
    /// Number of fp_cache misses (required full canonical computation).
    pub(super) fp_cache_misses: u64,
    /// Number of states folded into existing canonical representatives.
    /// Incremented when a new state's canonical fingerprint matches an already-seen state.
    pub(super) states_folded: u64,
    /// Number of times the fp_cache was evicted (cleared) due to exceeding the soft cap.
    ///
    /// Part of #4080: OOM safety — bounded symmetry fp_cache.
    pub(super) fp_cache_evictions: u64,
    /// Names of model value set constants that contribute to symmetry groups.
    pub(super) group_names: Vec<String>,
    /// Whether symmetry was auto-detected from config model value sets.
    pub(super) auto_detected: bool,
    /// Per-checker override for automatic symmetry detection.
    ///
    /// `None` (default) falls back to the `TY_AUTO_SYMMETRY` environment
    /// variable. Tests and embedders MUST use this override instead of
    /// mutating the process environment: `std::env::set_var` is process-global,
    /// so toggling `TY_AUTO_SYMMETRY` in one test silently enables symmetry
    /// reduction in concurrently-running checkers (observed as flaky
    /// state-count assertions in the symmetry test suite).
    pub(super) auto_symmetry_override: Option<bool>,
}

/// Part of #2752: Shared periodic liveness state — re-export from crate root
/// so sequential and parallel paths use identical gating logic.
pub(super) use crate::periodic_liveness::PeriodicLivenessState;

/// Debug instrumentation state (only active in debug builds).
pub(super) struct DebugDiagnostics {
    /// Map TLC FP -> internal FP to detect dedup mismatches.
    pub(super) seen_tlc_fp_dedup: Option<FxHashMap<u64, Fingerprint>>,
    /// Number of times a TLC fingerprint was seen with a different internal FP.
    pub(super) seen_tlc_fp_dedup_collisions: u64,
    /// Maximum collisions to print.
    pub(super) seen_tlc_fp_dedup_collision_limit: usize,
    /// Number of seen states containing lazy values.
    #[cfg(debug_assertions)]
    pub(super) lazy_values_in_state_states: u64,
    /// Number of lazy values observed across all recorded states.
    #[cfg(debug_assertions)]
    pub(super) lazy_values_in_state_values: u64,
    /// Maximum lazy-value state lines to print.
    #[cfg(debug_assertions)]
    pub(super) lazy_values_in_state_log_limit: usize,
    /// Map internal FP -> first TLC FP to detect internal FP collisions.
    pub(super) internal_fp_collision: Option<FxHashMap<Fingerprint, u64>>,
    /// Number of internal FP collisions detected.
    pub(super) internal_fp_collisions: u64,
    /// Maximum internal FP collisions to print.
    pub(super) internal_fp_collision_limit: usize,
}

/// Runtime hooks and progress reporting state.
pub(super) struct RuntimeHooksState {
    pub(super) init_progress_callback: Option<InitProgressCallback>,
    pub(super) progress_callback: Option<ProgressCallback>,
    pub(super) progress_interval: usize,
    pub(super) last_capacity_status: CapacityStatus,
    /// When the memory/disk pressure checks last ran (2026-07 OOM audit).
    ///
    /// The sequential BFS memory poll was gated on a STATE-COUNT interval
    /// only; slow-expanding but wide states (seconds per dequeue, many MB of
    /// successors each) could go unpolled for minutes. This timestamp drives
    /// the wall-clock backstop in `maybe_report_progress`, porting the
    /// parallel collector's 1-second poll cadence (finalize/collect.rs).
    pub(super) last_memory_check: Instant,
}

/// Coverage collection state.
pub(super) struct CoverageState {
    /// Whether to collect per-action coverage statistics.
    ///
    /// V2 vacuity gate (TRUST_VACUITY_GATE §1.A): this is forced ON by default
    /// so the per-action fired-bitset is maintained unconditionally and
    /// dead-action detection is always available. Display of the full coverage
    /// report is gated separately by [`CoverageState::display`].
    pub(super) collect: bool,
    /// Whether to print the full per-action coverage REPORT at end of run.
    ///
    /// Set by `--coverage` / `--coverage-guided`. When false, coverage is still
    /// tracked (for the V2 dead-action WARNING) but the verbose report is
    /// suppressed and `stats.coverage` is cleared at finalize for output parity.
    pub(super) display: bool,
    /// Whether `collect` was selected specifically by the default V2
    /// dead-action warning policy.
    ///
    /// Unlike explicit coverage requests, this non-correctness tracking may
    /// yield to an armed Value action VM after the concrete BFS route has
    /// proved that no action-boundary consumer is otherwise required.
    pub(super) default_dead_action_tracking: bool,
    /// Whether default dead-action coverage was skipped because the native
    /// fast path looked structurally viable (`set_default_dead_action_coverage`
    /// / `native_fast_path_coverage_skippable`).
    ///
    /// Consumed by `auto_select_post_compile_trust_cg_gate`: when the
    /// post-compile gate abandons native (partial action coverage, no admitted
    /// fused level) the perf rationale for the skip never materialized, so the
    /// gate re-enables collection before the BFS loop takes any transition —
    /// restoring the default V2 dead-action WARNING for interpreter runs.
    pub(super) native_fast_path_skipped: bool,
    /// Cached detected actions (including expressions) for coverage collection.
    ///
    /// SOUNDNESS: `Arc`-shared so per-state handles in the per-action successor
    /// paths reference the SAME run-stable AST allocations. The unified
    /// enumerator memoizes per-call-site results in pointer-keyed caches
    /// (subst_cache, const_domain_cache, expr_analysis, branch replay); feeding
    /// it per-state deep clones lets freed node addresses be reused across
    /// actions/states, replaying stale entries (wrong INSTANCE argument
    /// substitutions → mis-bound parameters, false violations, false deadlocks).
    pub(super) actions: Arc<Vec<DetectedAction>>,
    /// Retired action ASTs (replaced mid-run, e.g. native-fused auto-POR
    /// release). Kept alive so pointer-keyed enumeration caches holding
    /// entries for these nodes can never alias newly allocated expressions.
    pub(super) retired_actions: Vec<Arc<Vec<DetectedAction>>>,
    /// Whether to use coverage-guided exploration (priority frontier).
    pub(super) coverage_guided: bool,
    /// Coverage tracker for guided exploration (populated when coverage_guided=true).
    pub(super) tracker: Option<crate::coverage::guided::CoverageTracker>,
    /// Mix ratio for coverage-guided frontier (every N pops, one is priority-guided).
    pub(super) mix_ratio: usize,
}

/// Liveness checking cache state.
pub(super) struct LivenessCacheState {
    /// Cached successors from BFS (fingerprint -> list of successor fingerprints).
    /// Used for liveness checking to avoid regenerating transitions.
    /// Part of #3176: now uses `SuccessorGraph` dispatch enum supporting
    /// both in-memory HashMap and disk-backed storage.
    pub(super) successors: SuccessorGraph,
    /// Concrete successor witnesses keyed by canonical source fingerprint.
    ///
    /// Under SYMMETRY, liveness action checks must evaluate against the concrete
    /// successor states generated during BFS, not against the reduced
    /// representative state recovered later from `seen`.
    pub(super) successor_witnesses: SuccessorWitnessCache,
    /// 2026-07 memory audit: raw-fingerprint interning pool for the symmetry
    /// successor witnesses above.
    ///
    /// Without interning, `successor_witnesses` retained one independently
    /// allocated concrete `ArrayState` per generated TRANSITION for the whole
    /// run (BufferedRandomAccessFile: 248,697 retained states for only 6,376
    /// distinct ones — ~200 MB of duplicate `Value` trees). Keyed by the raw
    /// (pre-canonicalization) content fingerprint, so every duplicate concrete
    /// successor shares one allocation. Equal raw fingerprints are trusted as
    /// equal values — exactly the trust the BFS dedup places in fp64
    /// everywhere else (a collision here substitutes a value-equal witness
    /// candidate with probability 1-2^-64, same policy as
    /// `SUBSCRIPT_VALUE_CACHE`). Kill switch: `TY_NO_WITNESS_INTERN=1`.
    pub(super) witness_intern: FpHashMap<ArrayState>,
    /// Fairness constraints extracted from SPEC formula
    pub(super) fairness: Vec<FairnessConstraint>,
    /// Deduplicated fairness-derived state leaves keyed by stable fairness tags.
    ///
    /// Part of #3065: these are evaluated during BFS and copied into the
    /// existing cross-property cache before post-BFS liveness checking.
    pub(super) fairness_state_checks: Vec<LiveExpr>,
    /// Deduplicated fairness-derived action leaves keyed by stable fairness tags.
    pub(super) fairness_action_checks: Vec<LiveExpr>,
    /// Maximum fairness tag produced by fairness-first LiveExpr conversion.
    pub(super) fairness_max_tag: u32,
    /// Part of #3100: Reverse-indexed action provenance: split_action index → [tags].
    ///
    /// `action_provenance_tags[action_idx]` lists every fairness `ActionPred` tag
    /// proven true whenever split action `action_idx` fires. Built during liveness
    /// preparation by matching ActionPred hints against all split_action_meta entries
    /// (not just the first match). Keyed by action_idx for O(1) lookup in the BFS
    /// successor loop — avoids scanning all fairness leaves per successor.
    pub(super) action_provenance_tags: Vec<Vec<u32>>,
    /// Split-action provenance tags that are safe for runtime prepopulation.
    ///
    /// INSTANCE-qualified ModuleRef hints are still tracked in
    /// `action_provenance_tags` for diagnostics/tests, but they are excluded here
    /// until split-action matching can prove reference-equivalent semantics (#3161).
    pub(super) action_fast_path_provenance_tags: Vec<Vec<u32>>,
    /// Part of #3100: ENABLED-based action skip groups (TLC's WF disjunction short-circuit).
    ///
    /// When `ENABLED(<<A>>_vars)` is false for state s, the conjunction
    /// `<<A>>_vars = ActionPred ∧ StateChanged` is false for ALL transitions from s.
    /// Each group maps an ENABLED state-leaf tag to the action-leaf tags (ActionPred +
    /// StateChanged) from the same WF/SF fairness constraint. Pre-populating these as
    /// false when ENABLED is false avoids expensive AST/compiled action evaluation.
    ///
    /// TLC equivalent: `LNAction.eval()` short-circuits on subscript check; combined
    /// with WF disjunction ordering, TLC skips action body evaluation when the action
    /// is not enabled.
    pub(super) enabled_action_groups: Vec<EnabledActionGroup>,
    /// Part of #4179: Set of action leaf tags covered by at least one
    /// `action_provenance_tags[*]` entry. Tags in this set can be selectively
    /// evaluated based on the split_action that produced a given successor,
    /// avoiding evaluation of action predicates that cannot be true.
    // #4179 provenance scaffolding: written-once via struct literal but not yet read.
    #[allow(dead_code)]
    pub(super) provenance_covered_tags: rustc_hash::FxHashSet<u32>,
    /// Part of #4179: Action leaves whose tags are NOT in `provenance_covered_tags`.
    /// These must always be evaluated for every transition regardless of which
    /// split_action produced the successor, because provenance cannot prove they
    /// are false. Built during `prepare_inline_fairness_cache`.
    // #4179 provenance scaffolding: written-once via struct literal but not yet read.
    #[allow(dead_code)]
    pub(super) provenance_uncovered_action_leaves: Vec<LiveExpr>,
    /// Part of #3100: Subscript-action pairs for TLC's LNAction short-circuit.
    /// Maps `StateChanged(v)` tags to paired `ActionPred(A)` tags from `<<A>>_v`.
    pub(super) subscript_action_pairs: Vec<SubscriptActionPair>,
    /// Part of #3100: ENABLED provenance bypass — ENABLED tag → split_action
    /// indices. If any successor's action_tag matches, ENABLED is true.
    pub(super) enabled_provenance: Vec<EnabledProvenanceEntry>,
    /// Bitmask-only inline state results recorded during BFS.
    /// Bit `tag` set when `(fp, tag) → true`. Fingerprint presence means
    /// all fairness state tags have been evaluated for that state.
    /// Part of #3177: backed by `StateBitmaskMap` (in-memory or disk).
    pub(super) inline_state_bitmasks: StateBitmaskMap,
    /// Bitmask-only inline action results recorded during BFS.
    /// Bit `tag` set when `(from_fp, to_fp, tag) → true`. Key presence means
    /// all fairness action tags have been evaluated for that transition.
    /// Part of #3177: backed by `ActionBitmaskMap` (in-memory or disk).
    pub(super) inline_action_bitmasks: ActionBitmaskMap,
    /// Property-scoped inline liveness plans and recorded leaf results.
    pub(super) inline_property_plans: Vec<InlineLivenessPropertyPlan>,
    /// Cache successors for liveness (active when properties exist and liveness not skipped).
    /// Part of #3175: no longer requires store_full_states.
    pub(super) cache_for_liveness: bool,
    /// Testing-only override: when set, the reachable state graph is captured into
    /// `successors` regardless of whether the spec declares liveness properties, so
    /// cross-backend state-graph parity tests can compare the full reachable graph.
    ///
    /// This is set exclusively by `enable_state_graph_capture_for_testing()` and is
    /// OR-ed into `cache_for_liveness` by `refresh_liveness_cache_requirement()` so
    /// the capture survives the liveness-requirement recompute during check setup.
    /// Gated behind the `testing` feature so it adds zero production overhead.
    #[cfg(feature = "testing")]
    pub(super) force_capture_for_testing: bool,
    /// Initial states cached during BFS for post-BFS liveness checking.
    /// Part of #3175: populated regardless of store_full_states mode.
    /// Small: typically 1-10 states.
    pub(super) init_states: Vec<(Fingerprint, ArrayState)>,
    /// Part of #3210: Cached fp-only state replay result. Populated on first
    /// call to `build_fp_only_liveness_state_cache` and reused across properties
    /// to avoid O(S×D) per-state trace reconstruction per property.
    #[allow(clippy::type_complexity)]
    pub(super) fp_only_replay_cache: Option<(
        Arc<FxHashMap<Fingerprint, ArrayState>>,
        Arc<FxHashMap<Fingerprint, Fingerprint>>,
    )>,
    /// Part of #liveness-bfs-state-seed: `fp → ArrayState` pairs captured
    /// opportunistically at each inline-liveness state completion, keyed by
    /// the same BFS fingerprints the successor graph uses. Consumed (drained)
    /// by `build_fp_only_liveness_state_cache`, replacing its post-BFS
    /// Next-relation replay for every seeded state. Populated only when a
    /// tableau-carrying inline property plan guarantees the fp-only phase
    /// will build that cache anyway (see `maybe_seed_fp_only_state_cache`),
    /// so peak memory is the same map the replay would have built — just
    /// earlier. Capped by `TY_REPLAY_CACHE_MAX` like the replay itself.
    pub(super) bfs_seeded_states: FxHashMap<Fingerprint, ArrayState>,
    /// Large-liveness memory guard (`TY_LIVENESS_REGEN_BUDGET_MB`): set to `true`
    /// mid-BFS by [`maybe_trip_liveness_regen_budget`] when the cached liveness
    /// structures (successor graph + inline bitmasks + seeded states) exceed the
    /// configured byte budget. Once set, BFS stops caching (the caches are
    /// dropped to free memory) and the post-BFS liveness pass REGENERATES system
    /// successors on demand instead of reading the cached graph — matching TLC's
    /// memory/time tradeoff. Never true when the budget is disabled (`0`), so the
    /// historical always-cached behavior is available through the explicit
    /// `TY_LIVENESS_REGEN_BUDGET_MB=0` kill switch.
    pub(super) regenerate_on_the_fly: bool,
}

/// State storage: seen fingerprints and optional canonical payload witnesses.
pub(super) struct StateStorage {
    /// Seen states keyed by fingerprint.
    ///
    /// Full-state mode uses this for trace reconstruction. Fingerprint-indexed
    /// modes may also retain canonical payload witnesses so shared collision
    /// admission can prove a duplicate is the same state and fail closed when it
    /// is not.
    /// Part of #4128: Identity hash — fingerprints are pre-hashed.
    pub(super) seen: FpHashMap<ArrayState>,
    /// Seen fingerprints.
    ///
    /// Uses the shared FingerprintSet trait, which supports both in-memory
    /// HashSet and memory-mapped storage.
    pub(super) seen_fps: Arc<dyn FingerprintSet>,
    /// Whether to force full states as the primary trace-reconstruction store.
    ///
    /// When false, the checker still may keep selected canonical payload
    /// witnesses for collision-checked dedup admission.
    pub(super) store_full_states: bool,
    /// Exact compact payload witnesses for compiled-flat fingerprint-only lanes.
    ///
    /// Native fused BFS can admit flat i64 buffers without retaining evaluator
    /// states. These witnesses authorize true duplicates under the shared
    /// canonical-payload-equality fingerprint policy and fail closed on mismatch.
    pub(super) compiled_flat_payload_witnesses: FingerprintPayloadWitnesses,
}

/// Trace reconstruction state: depths and disk-based trace storage.
///
/// Part of #3178: the `parents` FxHashMap has been removed. Parent
/// relationships are now stored in the trace file for both full-state
/// and fp-only modes, saving ~16 bytes per state on the BFS hot path.
pub(super) struct TraceState {
    /// Depth tracking for each state (fingerprint -> depth)
    /// Part of #4128: Identity hash — fingerprints are pre-hashed.
    pub(super) depths: FpHashMap<usize>,
    /// Whether to auto-create a temp trace file (both full-state and fp-only modes)
    /// Default: true. Set to false for --no-trace mode (no trace reconstruction at all)
    pub(super) auto_create_trace_file: bool,
    /// Disk-based trace file for large state space exploration
    /// When enabled, stores (predecessor_loc, fingerprint) pairs on disk for trace reconstruction
    pub(super) trace_file: Option<TraceFile>,
    /// Mapping from fingerprint to trace file location
    /// Uses TraceLocationsStorage for scalable (mmap) or in-memory storage
    pub(super) trace_locs: TraceLocationsStorage,
    /// Whether trace I/O errors have occurred, meaning counterexample traces may be incomplete
    pub(super) trace_degraded: bool,
    /// Trace file location of the current parent state, set during BFS dequeue.
    ///
    /// Part of #2881 Step 3: eliminates per-successor `trace_locs` HashMap reads
    /// by carrying the parent's trace_loc from the queue entry. Set once per
    /// dequeued parent state, read by all successor admissions in that iteration.
    pub(super) current_parent_trace_loc: Option<u64>,
    /// Trace file location of the last successfully inserted state.
    ///
    /// Part of #2881 Step 3: used by `admit_successor` to include the trace_loc
    /// on queue entries without changing return types. Set by
    /// `mark_state_seen_checked` / `mark_state_seen_fp_only_checked` after a
    /// successful trace file write.
    pub(super) last_inserted_trace_loc: u64,
    /// When true, skip `trace_locs.insert` during BFS state admission.
    ///
    /// Part of #2881 Step 3: eliminates the last per-state HashMap write on
    /// the BFS hot path. The trace index is built lazily from the trace file
    /// via `ensure_trace_index_built()` when reconstruction is needed (cold
    /// path only). Default: false (eager inserts for backward compatibility
    /// with tests and callers that don't use queue entry threading).
    pub(super) lazy_trace_index: bool,
    /// Cached Init operator name (for trace reconstruction from fingerprints)
    pub(super) cached_init_name: Option<String>,
    /// Cached Next operator name (for trace reconstruction from fingerprints)
    pub(super) cached_next_name: Option<String>,
    /// Cached resolved Next operator name (avoids per-state resolve_op_name + String alloc)
    pub(super) cached_resolved_next_name: Option<String>,
}

impl TraceState {
    /// Build the trace location index from the trace file.
    ///
    /// Part of #2881 Step 3: when `lazy_trace_index` is true, the BFS hot
    /// path skips `trace_locs.insert` to avoid per-state HashMap writes.
    /// This method builds the index on demand by scanning the trace file.
    /// Called before any operation that needs fingerprint-to-offset lookups
    /// (error trace reconstruction, resume queue building).
    ///
    /// No-op when the index is already populated or no trace file exists.
    pub(super) fn ensure_trace_index_built(&mut self) {
        if !self.lazy_trace_index {
            return; // Eager mode — index already populated during BFS
        }

        // Check if the index already covers all trace file records.
        // Initial states may have been inserted eagerly before lazy mode was
        // enabled, so trace_locs.len() > 0 does not mean the index is complete.
        let file_records = match self.trace_file {
            Some(ref f) => f.record_count() as usize,
            None => return, // No trace file, nothing to build
        };

        if self.trace_locs.len() >= file_records {
            return; // Index already complete
        }

        if let Some(ref mut trace_file) = self.trace_file {
            match trace_file.read_all_records() {
                Ok(records) => {
                    for (fp, offset) in records {
                        self.trace_locs.insert(fp, offset);
                    }
                }
                Err(_) => {
                    // Trace file I/O failure; the caller (reconstruct_trace_from_file)
                    // will report a warning when the fingerprint lookup fails.
                    self.trace_degraded = true;
                }
            }
        }
    }
}

/// Part of #3100: Metadata from `ActionInstance` stored for action provenance matching.
///
/// Captures the action name and binding values from each split action instance,
/// enabling the liveness evaluator to match `ActionPred` expressions against BFS
/// actions by comparing operator names and binding values.
#[derive(Debug, Clone)]
pub(super) struct ActionInstanceMeta {
    /// Operator name of the action (e.g., "RcvMsg" for `RcvMsg(p)`).
    pub(super) name: Option<String>,
    /// Bound variable values from EXISTS expansion (e.g., `[(p, P1)]` for `RcvMsg(p)` with p=P1).
    pub(super) bindings: Vec<(Arc<str>, crate::Value)>,
    /// Base operator formal values in declaration order, when action splitting
    /// specialized a user-defined action application.
    pub(super) formal_bindings: Vec<(Arc<str>, crate::Value)>,
    /// The expression body for this split action disjunct.
    ///
    /// Stored so that `try_jit_monolithic_successors` can fall back to per-action
    /// interpreter evaluation for actions that are not JIT-compiled, enabling
    /// hybrid JIT/interpreter dispatch instead of all-or-nothing.
    ///
    /// Part of #3968: per-action hybrid JIT dispatch.
    #[allow(dead_code)]
    pub(super) expr: Option<tla_core::Spanned<tla_core::ast::Expr>>,
}

/// Pre-compiled spec artifacts: invariants, Next action, VIEW, and TLCExt!Trace detection.
pub(super) struct CompiledSpec {
    /// Part of #3100: Action instance metadata from split-action discovery.
    ///
    /// Each entry corresponds to the same index in the discovered split-action
    /// list. Stores the bindings from the original `ActionInstance`, used to
    /// build the `action_provenance_tags` during liveness preparation.
    pub(super) split_action_meta: Option<Vec<ActionInstanceMeta>>,
    /// Complete unpruned lexical binding chains parallel to
    /// `split_action_meta`. Kept only until Value Action VM setup so optional
    /// guard certificates and canonical replay preserve exact shadowing.
    pub(super) split_action_complete_bindings: Option<Vec<Vec<(Arc<str>, crate::Value)>>>,
    /// Cached VIEW operator name (for state abstraction in fingerprinting)
    /// When set, fingerprinting uses this operator's value instead of full state
    pub(super) cached_view_name: Option<String>,
    /// Whether the spec references TLCExt!Trace (conservative detection).
    /// When true, trace reconstruction is performed before invariant checks.
    pub(super) uses_trace: bool,
    /// Properties of the form `[]P` where P is a state-level predicate that were
    /// promoted to invariant checking during BFS (#2332). These are skipped in
    /// post-BFS liveness checking since they're already checked during exploration.
    pub(super) promoted_property_invariants: Vec<String>,
    /// Property names whose state-level terms are checked during BFS.
    ///
    /// This includes mixed properties with a remaining temporal tail and is
    /// used only for reporting `PropertyViolation` instead of
    /// `InvariantViolation` when the state-level part fails.
    pub(super) state_property_violation_names: Vec<String>,
    /// Eval-based implied actions for ModuleRef/INSTANCE properties (#2983).
    /// These use the general evaluator BFS-inline instead of compiled guards,
    /// because priming through INSTANCE WITH substitutions is buggy in the
    /// compiled guard path. Checked for ALL transitions during BFS.
    pub(super) eval_implied_actions: Vec<crate::checker_ops::EvalImpliedActionTerm>,
    /// Subset of `eval_implied_actions` whose bodies are native-capable action
    /// predicates. Trust-CG native fused BFS may consume these only when every
    /// term has backend coverage; eval-only terms still force fail-closed
    /// interpreter handling.
    pub(super) native_implied_actions: Vec<crate::checker_ops::NativeImpliedActionTerm>,
    /// Names of properties that had action-level always-terms extracted for
    /// BFS-phase implied action checking (#2670). These are skipped in
    /// post-BFS liveness checking since they're already checked during BFS.
    pub(super) promoted_implied_action_properties: Vec<String>,
    /// Eval-based state invariants for ENABLED-containing state-level terms (#3113).
    /// Checked for unseen states only via eval_entry() during BFS.
    pub(super) eval_state_invariants: Vec<(String, tla_core::Spanned<tla_core::ast::Expr>)>,
    /// Pre-computed variable dependency analysis for ACTION_CONSTRAINTs.
    /// Used to skip constraint evaluation when referenced variables are unchanged.
    pub(super) action_constraint_analysis: Option<crate::checker_ops::ActionConstraintAnalysis>,
    /// Setup-time certificate that every configured state `CONSTRAINT` is a
    /// pure, context-free predicate of the unprimed state. An exact payload
    /// match against a previously admitted state may reuse its successful
    /// constraint verdict only when this is true.
    pub(super) state_constraints_reusable_on_exact_duplicate: bool,
    /// Init predicates from PROPERTY entries: non-Always state/constant-level
    /// terms checked against initial states only. Part of #2834.
    pub(super) property_init_predicates: Vec<(String, tla_core::Spanned<tla_core::ast::Expr>)>,
    /// PlusCal `pc`-dispatch table for skipping irrelevant actions.
    ///
    /// When the spec follows the PlusCal pattern where all disjuncts of Next
    /// are guarded by `pc = "label"`, this table maps pc values to action
    /// indices, allowing the BFS loop to skip actions whose pc guard cannot
    /// match the current state.
    pub(super) pc_dispatch: Option<crate::checker_ops::pc_dispatch::PcDispatchTable>,
    /// Variable index of `pc` for guard hoisting, independent of the dispatch table.
    ///
    /// For single-process PlusCal specs, this is set from `pc_dispatch.pc_var_idx`.
    /// For multi-process PlusCal specs (where the full dispatch table can't be built
    /// because `pc` is a function), this is set directly by detecting that the spec
    /// uses `pc[self] = "label"` guard patterns. This allows the unified enumerator
    /// to skip Or branches with non-matching pc guards even for multi-process specs.
    ///
    /// Part of #3805: multi-process PlusCal guard hoisting.
    pub(super) pc_var_idx: Option<tla_core::VarIndex>,
    /// Whether the spec's AST contains expressions that can produce lazy values
    /// at runtime (`FuncDef`, `SetFilter`, `Lambda`). When `false`, the
    /// per-successor `has_lazy_state_value` scan in `materialize_array_state`
    /// and `materialize_diff_changes` is skipped entirely.
    ///
    /// Part of #4053: Skip `has_lazy_state_value` for non-lazy specs.
    pub(super) spec_may_produce_lazy: bool,
}

/// Exploration control: limits, deadlock checking, and error continuation.
pub(super) struct ExplorationControl {
    /// Whether to check for deadlocks
    pub(super) check_deadlock: bool,
    /// Force PURE explicit-state BFS: skip the symbolic inductive-safety
    /// certificate shortcut (and any other symbolic-verdict deferral) so the run
    /// genuinely ENUMERATES reachable states. Set by the certifying-verification
    /// eval oracle, whose whole purpose is an engine-diverse explicit re-check —
    /// it must not be short-circuited by the same symbolic path that produced the
    /// certificate. Defaults to `false` (normal cooperative behavior).
    pub(super) force_explicit_bfs: bool,
    /// Whether stuttering transitions are allowed (`[A]_v` = true, `<<A>>_v` = false).
    ///
    /// When true (the common `[A]_v` case), the liveness checker adds self-loop edges
    /// so that stuttering behaviors are visible to SCC analysis. Without these edges,
    /// liveness properties like `[]<>(x=3)` that are violated by infinite stuttering
    /// would be falsely reported as satisfied.
    ///
    /// Defaults to `true` (stuttering allowed), matching the most common TLA+ pattern.
    pub(super) stuttering_allowed: bool,
    /// Continue exploring after invariant/property violations (like TLC's -continue flag)
    pub(super) continue_on_error: bool,
    /// First invariant violation found (used in continue_on_error mode).
    /// Stores (invariant_name, violating_state_fingerprint).
    /// When continue_on_error is true, we record the first violation here and continue
    /// exploring until the full state space is exhausted, then report this violation.
    pub(super) first_violation: Option<(String, Fingerprint)>,
    /// First action-level PROPERTY violation found (used in continue_on_error mode).
    pub(super) first_action_property_violation: Option<(String, Fingerprint)>,
    /// Maximum states to explore (None = unlimited)
    pub(super) max_states: Option<usize>,
    /// Maximum BFS depth (None = unlimited)
    pub(super) max_depth: Option<usize>,
    /// Part of #3: Wall-clock deadline for BFS exploration (None = no time backstop).
    ///
    /// When set, the unified BFS worker loop polls this every
    /// `DEADLINE_CHECK_INTERVAL` states and stops cleanly with a partial
    /// `LimitReached { Time }` result once it elapses. This is the time backstop
    /// that prevents an unbounded spec from exploring forever / OOMing.
    pub(super) deadline: Option<Instant>,
    /// Part of #2751: Optional memory limit for threshold-triggered stop.
    pub(super) memory_policy: Option<crate::memory::MemoryPolicy>,
    /// Part of #3282: Optional disk usage limit in bytes for disk-backed storage.
    pub(super) disk_limit_bytes: Option<usize>,
    /// Part of #4080: Hard cap on estimated internal memory (bytes).
    ///
    /// Independent of RSS-based `memory_policy`. Triggers BFS stop when
    /// internal stores (FP set + seen + depths + queue) exceed this limit.
    /// Default: derived from `memory_policy` limit * 0.75.
    /// Override: `TY_INTERNAL_MEMORY_LIMIT` env var (bytes). 0 = disabled.
    pub(super) internal_memory_limit: Option<usize>,
}

/// The model checker
pub struct ModelChecker<'a> {
    /// Per-instance AUTO-selector structural-veto latch (Stage 3 of the unified-backend
    /// migration; replaces the former process-global `TRUST_CG_STRUCTURAL_VETO` static).
    /// Once the AUTO selector decides native won't help this run, this latches and every
    /// `should_use_trust_cg` / `is_enabled` gate routes to the interpreter. Per-instance
    /// scope makes daemon/library/server reuse safe (no cross-run coupling).
    pub(super) trust_cg_structural_veto: std::sync::atomic::AtomicBool,
    /// Configuration
    pub(super) config: &'a Config,
    /// Module metadata (names, paths, setup errors, assumes)
    pub(super) module: ModuleState,
    /// Evaluation context
    pub(super) ctx: EvalCtx,
    /// State storage (seen states, fingerprints, storage mode)
    pub(super) state_storage: StateStorage,
    /// Trace reconstruction (parents, depths, trace file, cached names)
    pub(super) trace: TraceState,
    /// Pre-compiled spec artifacts (invariants, Next action, VIEW, Trace detection)
    pub(super) compiled: CompiledSpec,
    /// Exploration limits and error handling
    pub(super) exploration: ExplorationControl,
    /// Statistics
    pub(super) stats: CheckStats,
    /// Per-run diagnostic counters (suppressed guard errors, implied-action
    /// telemetry). Owned by this checker instance and installed in a
    /// thread-local scope for the duration of each run so deep recording
    /// sites attribute counts to THIS run instead of a process-global
    /// counter that concurrent runs (cargo test) reset/steal.
    pub(super) run_diagnostics: std::sync::Arc<crate::run_diagnostics::RunDiagnostics>,
    /// Runtime hooks and progress reporting
    pub(super) hooks: RuntimeHooksState,
    /// Liveness checking cache
    pub(super) liveness_cache: LivenessCacheState,
    /// Part of #3225: computed once from storage/symmetry/VIEW instead of
    /// re-deriving the liveness mode matrix at every callsite.
    pub(super) liveness_mode: LivenessMode,
    /// Coverage collection
    pub(super) coverage: CoverageState,
    /// Symmetry reduction state (permutations, fast MVPerms, fingerprint cache)
    pub(super) symmetry: SymmetryState,
    /// Env-gated AST/TIR parity replay for selected named operators.
    pub(super) tir_parity: Option<super::tir_parity::TirParityState>,
    /// Exact projection-keyed TRUE cache for sequential named invariants.
    /// Parallel workers deliberately do not consume this first-slice cache.
    pub(super) invariant_verdict_cache: super::invariants::InvariantVerdictCache,
    /// Exact Boolean-verdict cache for the complete ordered state-constraint list.
    /// Misses retain the canonical bytecode/TIR/AST dispatch unchanged.
    pub(super) state_constraint_verdict_cache:
        super::invariants::StateConstraintVerdictCache,
    /// Part of #3578: Compiled bytecode for invariant operators (bytecode VM fast path).
    pub(super) bytecode: Option<tla_eval::bytecode_vm::CompiledBytecode>,
    /// Part of #3910: Compiled bytecode for next-state action operators (JIT next-state fast path).
    /// Separate from invariant bytecode because actions use StoreVar opcodes for primed variables.
    pub(super) action_bytecode: Option<tla_eval::bytecode_vm::CompiledBytecode>,
    /// Opt-in, fail-closed single-successor Value action VM dispatch.
    ///
    /// Its executable plan is certified only after action bytecode reaches its
    /// final transformed/split shape. Runtime errors and shadow mismatches
    /// route back through the canonical interpreter.
    pub(super) value_action_vm: super::value_action_vm::ValueActionVmDispatch,
    /// Lever 1 (#EWD998PCal): CONSTRAINT operators compiled once per run to
    /// bytecode, executed directly over the candidate `ArrayState` (no ctx
    /// binds, no per-check cache clears, no per-check `TirProgram` rebuild).
    /// Compiled even when TIR eval mode blocks the invariant bytecode compile
    /// (the per-check TIR constraint path is exactly the rebuild waste this
    /// replaces). `None` when there are no constraints, the VM is disabled,
    /// or compilation failed for any constraint op.
    pub(super) constraint_bytecode: Option<tla_eval::bytecode_vm::CompiledBytecode>,
    /// Fail-closed latch: set on the first VM error/non-boolean result during
    /// a constraint fast-path attempt, permanently routing this run back to
    /// the canonical interpreter/TIR constraint path (no repeated doomed VM
    /// executions). The fallback path re-evaluates ALL constraints for the
    /// triggering state, so behavior is byte-identical to the lever being off.
    pub(super) constraint_bytecode_disabled: bool,
    /// Part of #3582: fail-closed compiled invariant compatibility functions.
    /// Populated at run_prepare time when the compatibility path is enabled.
    pub(super) jit_cache: Option<JitInvariantCacheImpl>,
    /// Part of #3700: Reusable scalar-state scratch for JIT invariant checks.
    pub(super) jit_state_scratch: Vec<i64>,
    /// True when every configured invariant has a JIT-compiled native function.
    /// Enables the zero-allocation `check_all_compiled` fast path that skips
    /// the unchecked buffer entirely.
    pub(super) jit_all_compiled: bool,
    /// Pre-resolved JIT function pointers in invariant order. Eliminates
    /// per-invariant HashMap lookups in the hot path.
    pub(super) jit_resolved_fns: Option<Vec<tla_jit_abi::JitInvariantFn>>,
    /// Part of #3700: Sequential JIT profile hits (fully handled by JIT).
    pub(super) jit_hits: usize,
    /// Part of #3700: Sequential JIT profile misses (fell back after JIT attempt).
    pub(super) jit_misses: usize,
    /// Part of #3935: Per-invariant JIT dispatch hits.
    pub(super) jit_hit: usize,
    /// Part of #3935: Per-invariant JIT dispatch fallbacks.
    pub(super) jit_fallback: usize,
    /// Part of #3935: Per-invariant JIT dispatch misses (no native invariant compiled).
    pub(super) jit_not_compiled: usize,
    /// Part of #3935: Total per-invariant JIT dispatch attempts.
    pub(super) total_invariant_evals: usize,
    /// Number of fully JIT-covered invariant evaluations cross-checked via the interpreter.
    pub(super) jit_verify_checked: usize,
    /// Number of JIT/interpreter invariant mismatches observed during cross-checking.
    pub(super) jit_verify_mismatches: usize,
    /// Tiered JIT compilation manager for HotSpot-style action promotion.
    ///
    /// Tracks per-action compilation tiers (Interpreter → Tier1 → Tier2) and
    /// makes promotion decisions based on evaluation frequency. Initialized
    /// in `prepare_bfs_common` after action splitting discovers action count.
    ///
    /// Part of #3850: tiered JIT wiring into eval hot path.
    pub(super) tier_manager: Option<TierManagerImpl>,
    /// Per-action evaluation counts for tiered JIT promotion decisions.
    ///
    /// Lightweight Vec<u64> indexed by action_id. Updated during successor
    /// generation for both monolithic and split-action paths. Does not
    /// depend on cooperative/ay mode -- works for all BFS modes.
    ///
    /// Part of #3850: tiered JIT wiring into eval hot path.
    pub(super) action_eval_counts: Vec<u64>,
    /// Per-action successor totals for branching factor computation.
    ///
    /// Part of #3850: tiered JIT wiring into eval hot path.
    pub(super) action_succ_totals: Vec<u64>,
    /// Accumulated tier promotion events for `--show-tiers` report.
    ///
    /// Appended by `check_tier_promotions()` at progress intervals.
    /// Read during `run_finalize` to produce the per-action tier report.
    ///
    /// Part of #3910: Wire TierManager into BFS loop.
    pub(super) tier_promotion_history: Vec<tla_jit_abi::TierPromotion>,
    /// Reusable scratch buffer for type profiling during BFS.
    ///
    /// Sized to state variable count. Populated with `classify_value` results
    /// for each explored state and passed to `TierManager::observe_state_types`.
    /// Avoids per-state allocation.
    ///
    /// Part of #3989: speculative type specialization.
    pub(super) type_profile_scratch: Vec<tla_jit_abi::SpecType>,
    /// JIT-compiled next-state action cache for sequential BFS.
    ///
    /// Built lazily on first Tier 1 promotion (not at startup) to avoid
    /// compilation overhead for small specs that never cross the threshold.
    /// Once populated, the diff/full-state successor paths check this cache
    /// before falling back to the interpreter.
    ///
    /// Part of #3910: Wire TierManager into BFS loop.
    pub(super) jit_next_state_cache: Option<JitNextStateCacheImpl>,
    /// Next-state JIT dispatch counters for `--show-tiers` report.
    ///
    /// Part of #3910.
    pub(super) next_state_dispatch: tla_jit_abi::NextStateDispatchCounters,
    /// Compilation statistics from the last `JitNextStateCache::build_with_stats`
    /// call. Populated on first Tier 1 promotion. Included in the
    /// `--show-tiers` end-of-run report.
    ///
    /// Part of #3910: JIT compilation latency instrumentation.
    pub(super) jit_cache_build_stats: Option<tla_jit_abi::CacheBuildStats>,
    /// Pending async JIT compilation result.
    ///
    /// When a Tier 1 promotion fires, compilation is spawned on a background
    /// thread via `std::thread::spawn`. The sender half of a oneshot channel
    /// is moved into the thread; the receiver stays here. The BFS loop
    /// continues with the interpreter while the native cache builds. On each
    /// state, `poll_pending_jit_compilation` does a non-blocking `try_recv`.
    /// Once ready, the cache is moved to `jit_next_state_cache` and
    /// subsequent states use native code.
    ///
    /// Part of #3910: Async JIT compilation with interpreter warmup.
    pub(super) pending_jit_compilation:
        Option<std::sync::mpsc::Receiver<(JitNextStateCacheImpl, tla_jit_abi::CacheBuildStats)>>,
    /// Tier 2 recompilation controller for speculative type specialization.
    ///
    /// Manages the lifecycle of Tier 2 recompilation triggered by type profiling.
    /// When a Tier 2 promotion fires with a `SpecializationPlan`, this controller
    /// spawns a background thread to rebuild the JIT cache and polls for completion.
    ///
    /// Part of #3989: speculative type specialization.
    pub(super) recompilation_controller: RecompilationControllerImpl,
    /// Compound state layout inferred from the initial state.
    ///
    /// Populated by `upgrade_jit_cache_with_layout` after init state solving.
    /// Used by the async JIT compilation thread to enable native compound
    /// access (FuncApply, RecordGet, TupleGet) in next-state dispatch.
    ///
    /// Part of #3958: Enable native compound access in JIT next-state dispatch.
    pub(super) jit_state_layout: Option<tla_jit_abi::StateLayout>,
    /// Set to true when JIT successor generation must be permanently disabled
    /// for ALL actions. This is the global kill switch for catastrophic failures:
    /// (1) validation mismatch — JIT successor count differs from monolithic
    /// enumerator (#4011), (2) warmup gate decision, (3) compiled BFS step error,
    /// (4) async compilation thread disconnect.
    ///
    /// Per-action compilation failures use `jit_disabled_actions` instead.
    ///
    /// Part of #3968: per-action hybrid JIT dispatch.
    /// Part of #4011: also set on validation failure.
    /// Part of #4012: split from per-action to global-only.
    pub(super) jit_monolithic_disabled: bool,
    /// Per-action JIT disable flags. When `jit_disabled_actions[action_idx]` is
    /// true, that specific action has been disabled due to a JIT runtime error
    /// (compiled function returned an error). Other actions that compiled
    /// successfully continue to use JIT.
    ///
    /// Sized to action count when JIT cache is installed. Empty before that.
    ///
    /// Part of #4012: per-action JIT disable instead of global kill switch.
    pub(super) jit_disabled_actions: Vec<bool>,
    /// Cached flag: true when ALL split actions (including EXISTS-bound
    /// specializations) have JIT cache entries. Computed once when the
    /// JIT next-state cache is installed via `poll_pending_jit_compilation`.
    ///
    /// When false, `try_jit_monolithic_successors` bails out immediately
    /// instead of iterating through actions only to discover a cache miss
    /// partway through and abandoning the work.
    ///
    /// Part of EXISTS binding JIT dispatch: avoids per-state O(N) cache
    /// coverage re-checking.
    pub(super) jit_all_next_state_compiled: bool,
    /// Cached flag: true when at least one action is at Tier1+ compilation.
    /// Updated once when the JIT cache is installed and coverage is checked.
    /// Avoids per-state iteration over all actions in `jit_monolithic_ready()`.
    ///
    /// Part of #4030: Eliminate per-state O(N) action scan in JIT readiness check.
    pub(super) jit_has_any_promoted: bool,
    /// Cached flag: true when the current state's flat representation has
    /// no compound (non-int, non-bool) variables. When true, the JIT fused
    /// path can skip `clear_compound_scratch()` calls, saving a TLS access
    /// per action evaluation.
    ///
    /// Part of #4030: Skip compound scratch for all-scalar states.
    pub(super) jit_state_all_scalar: bool,
    /// When true, the BFS dedup pipeline uses xxh3 SIMD fingerprinting on
    /// flat i64 buffers instead of per-variable FP64 tree-walking. This is
    /// activated when ALL state variables are scalar (Int/Bool), no VIEW
    /// expression is configured, and no SYMMETRY reduction is active.
    ///
    /// CRITICAL: Once set to true at BFS start, ALL fingerprints (init states,
    /// successors from both JIT and interpreter paths) MUST use xxh3. Mixing
    /// xxh3 and FP64 fingerprints in the same dedup set causes silent state
    /// loss (different hash for the same logical state).
    ///
    /// Part of #3987: Compiled xxh3 fingerprinting for the BFS hot path.
    pub(super) jit_compiled_fp_active: bool,
    /// Debug-only seal that prevents fingerprint algorithm changes after BFS starts.
    ///
    /// Set to `true` at the start of `run_bfs_loop_core` / `run_compiled_bfs_loop`.
    /// `try_activate_compiled_fingerprinting` asserts this is still `false`,
    /// providing a structural guarantee that no mid-run algorithm switch is
    /// possible. This eliminates the class of bugs where flat (xxh3) and array
    /// (FP64) fingerprints coexist in the same seen set.
    ///
    /// Part of #4215: Fingerprint domain separation guarantee.
    #[cfg(debug_assertions)]
    pub(super) fp_algorithm_sealed: bool,
    /// Frozen BFS fingerprint domain, captured once at BFS start.
    ///
    /// `bfs_fingerprint_domain()` is derived from mutable checker state
    /// (notably `compiled_bfs_level.is_some()` via
    /// `state_constrained_native_fused_admission_active`). In AUTO lazy-compile
    /// mode the native fused level is installed MID-RUN once the distinct-state
    /// count crosses the threshold, which would otherwise flip the domain from
    /// `ArrayFp64` to `CompiledFlat` partway through the run — the init states
    /// and the pre-threshold successors were hashed FP64 while post-threshold
    /// successors would be hashed xxh3-flat, so a state reachable on both sides
    /// of the boundary lands in the dedup set twice and inflates the distinct
    /// count (observed EWD998Small 1520618 -> 1521489 under a forced low
    /// threshold). Freezing the domain at BFS start keeps every fingerprint in
    /// one domain for the whole run regardless of later lazy compilation.
    ///
    /// `None` until the first BFS loop entry captures it; after that
    /// `bfs_fingerprint_domain()` returns this value verbatim.
    pub(super) frozen_bfs_fingerprint_domain: Option<super::fingerprint::BfsFingerprintDomain>,
    /// Reusable scratch buffer for flat xxh3 fingerprinting.
    ///
    /// Avoids allocating a fresh `Vec<i64>` per successor in
    /// `array_state_fingerprint_xxh3`. Sized to `var_count` when
    /// `jit_compiled_fp_active` is set to true.
    ///
    /// Part of #3986: Eliminate per-state allocation in flat fingerprint path.
    pub(super) flat_fp_scratch: Vec<i64>,
    /// Remaining JIT validation cross-checks against the monolithic enumerator.
    ///
    /// When JIT produces successors for a state (all actions compiled), the
    /// result is cross-checked against the monolithic enumerator for the first
    /// N states. If successor counts differ, JIT is permanently disabled and
    /// a P0 warning is logged. After N successful validations, the
    /// double-computation is skipped.
    ///
    /// Part of #4011: JIT validation mode for fully-JIT states.
    pub(super) jit_validation_remaining: u32,
    /// Pre-computed JIT cache lookup keys for each split action.
    ///
    /// Computed once when the JIT cache is installed (in `poll_pending_jit_compilation`),
    /// avoiding per-state String allocation and clone overhead in the hot path.
    /// Each entry is the cache key for the corresponding action in split_action_meta:
    /// - Binding-free actions: the action name directly
    /// - EXISTS-bound actions: specialized_key(name, binding_values)
    ///
    /// Part of #4030: Eliminate per-state allocation in JIT hybrid dispatch.
    pub(super) jit_action_lookup_keys: Vec<String>,
    /// Inner EXISTS expansion keys for each split action (parallel to `jit_action_lookup_keys`).
    ///
    /// For most actions this is an empty Vec (no inner EXISTS). For actions whose
    /// inner EXISTS quantifiers were pre-expanded by the JIT, this contains all
    /// expansion keys (e.g., `["SendMsg__0", "SendMsg__1", "SendMsg__2"]`).
    ///
    /// During monolithic dispatch, when this is non-empty for an action, the
    /// dispatcher iterates ALL expansion keys (each produces at most one successor)
    /// instead of using the single key from `jit_action_lookup_keys`.
    ///
    /// Part of #4176: JIT EXISTS binding dispatch.
    pub(super) jit_inner_exists_keys: Vec<Vec<String>>,
    /// Reusable scratch buffer for JIT action output (successor state as i64[]).
    ///
    /// Avoids allocating a fresh Vec<i64> per action evaluation. Sized to
    /// state_var_count when the JIT cache is installed.
    ///
    /// Part of #4030: Eliminate per-action allocation in JIT dispatch.
    pub(super) jit_action_out_scratch: Vec<i64>,
    /// Adaptive performance monitoring for JIT dispatch.
    ///
    /// Tracks cumulative time spent in JIT vs interpreter paths for the first
    /// N states after JIT activation. If JIT is consistently slower, it is
    /// automatically disabled. Format: (jit_ns, interp_ns, states_sampled).
    ///
    /// Part of #4030: Adaptive JIT performance switch.
    pub(super) jit_perf_monitor: (u64, u64, u32),
    /// Cached TY_JIT_DIAG env var check. Set once when JIT cache is installed
    /// to avoid a per-state syscall in the hot path.
    ///
    /// Part of #4030: Eliminate per-state env var overhead.
    pub(super) jit_diag_enabled: bool,
    /// Compiled BFS step function for fully-JIT specs.
    ///
    /// When all actions AND all invariants are JIT-compiled, and the spec
    /// uses a fully-flat state representation (all variables are fixed-size
    /// i64 representable), this holds a `CompiledBfsStep` that performs
    /// the entire BFS inner loop (action dispatch, fingerprinting, dedup,
    /// invariant checking) in native compiled code.
    ///
    /// Built lazily after `poll_pending_jit_compilation` installs the
    /// `jit_next_state_cache` and coverage checks pass.
    ///
    /// Part of #4034: Wire CompiledBfsStep into model checker BFS loop.
    /// Boxed trait object so the BFS level loop remains backend-agnostic.
    /// trust-codegen provides the active implementation in `trust_cg_dispatch`; retired
    /// compiled backends are represented only by fail-closed compatibility
    /// shims.
    pub(super) compiled_bfs_step: Option<Box<dyn CompiledBfsStepTrait>>,
    /// Fused compiled BFS level function that processes entire frontiers in
    /// a single native call. Built lazily after `CompiledBfsStep` is
    /// available and the fused level function compiles successfully.
    ///
    /// When present, `run_compiled_bfs_loop()` uses this instead of the
    /// per-parent `CompiledBfsStep` path, eliminating Rust-to-JIT boundary
    /// crossings per parent.
    ///
    /// Part of #4171: End-to-end compiled BFS wiring.
    /// Boxed trait object for the same reason as `compiled_bfs_step` — the
    /// level loop binds to a backend-agnostic trait surface.
    pub(super) compiled_bfs_level: Option<Box<dyn CompiledBfsLevelTrait>>,
    /// Setup deferred the native fused `CompiledBfsLevel` compile.
    ///
    /// Compiling the fused parent-loop module is the single largest fixed
    /// setup cost on small runs (hundreds of milliseconds of trust-codegen
    /// regalloc on one large generated function), while the per-parent
    /// `CompiledBfsStep` path drives the compiled BFS loop with identical
    /// semantics. When this flag is set, setup intentionally skipped the
    /// fused-level build and `run_compiled_bfs_loop` promotes to the fused
    /// level at a level boundary once the run has proven large enough to
    /// amortize the compile (structural state-count trigger; see
    /// `should_defer_fused_level_build`). Runs that finish below the
    /// threshold never pay the compile at all.
    ///
    /// Soundness: the fused level and the per-parent step are
    /// verdict-equivalent per level — the loop already falls back from fused
    /// to step mid-run on runtime errors — so which levels run on which path
    /// cannot change state counts or verdicts.
    pub(super) deferred_fused_level_build: bool,
    /// The native fused level was built in *action-only* mode because the state
    /// space had not yet proven large enough to amortize fusing the invariant
    /// predicates into the generated parent loop (size-gate;
    /// `trust_cg_fused_invariant_min_states` / `TY_FUSED_INVARIANT_MIN_STATES`).
    ///
    /// While set, the compiled BFS loop checks invariants per successor in the
    /// interpreter (`check_successor_invariant`, exactly the default action-only
    /// path). At a level boundary, once the cumulative distinct-state count
    /// crosses the floor, `run_compiled_bfs_loop` rebuilds the level with the
    /// invariants fused and clears this flag — so small runs never pay the large
    /// invariant-fusion compile, and large runs still get native invariant
    /// checks for the bulk of the exploration.
    ///
    /// Soundness: action-only + interpreter invariant checks and the
    /// invariant-fused level are verdict-equivalent (invariant checking is the
    /// same predicate either way, native or interpreted), and the fingerprint
    /// domain is frozen at BFS start, so which path checks invariants cannot
    /// change state counts or verdicts. Gate is scoped to unconstrained runs;
    /// state-constrained runs always fuse eagerly (unchanged).
    pub(super) deferred_fused_invariant_build: bool,
    /// AUTO mode deferred the eager trust-cg native action-callout cache build.
    ///
    /// In AUTO engine-selection mode (`ty check` with no `--backend` flag) the
    /// trust-codegen compile (~0.5-0.6s of JIT regalloc) only pays off on large
    /// state spaces. Setup therefore skips the eager build and sets this flag;
    /// the interpreter Rust BFS loop runs while `trust_cg_cache`,
    /// `compiled_bfs_step`, and `compiled_bfs_level` all stay `None` (so
    /// `should_use_compiled_bfs()` declines the compiled loop). The per-parent
    /// BFS step (`maybe_trigger_trust_cg_lazy_compile`) builds the cache exactly
    /// once the distinct-state count reaches
    /// `trust_cg_lazy_compile_threshold()`, then clears this flag; small runs
    /// finish on the interpreter without ever paying the compile.
    ///
    /// Soundness: the interpreter path used before the build is the oracle, and
    /// the native per-action callout path installed after the build is
    /// parity-validated against it. Only the timing of the compile changes, never
    /// any state count, verdict, or trace. Forced `--backend trust-cg` never sets
    /// this flag (it keeps the eager build).
    pub(super) trust_cg_lazy_pending: bool,
    /// trust_cg-compiled native function cache for BFS dispatch.
    ///
    /// Current opt-in native backend: lowers TY bytecode through trust-ir into
    /// trust_cg-generated native code. Shares the same `extern "C"` ABI as the
    /// fail-closed compatibility layer.
    ///
    /// Activated by the `TY_TRUST_CG=1` environment variable.
    /// Built once at startup since trust-codegen compilation
    /// is an explicit opt-in.
    ///
    /// Part of #4118: Wire tla-trust_cg into tla-check BFS loop.
    pub(super) trust_cg_cache: Option<super::trust_cg_dispatch::TrustCgNativeCache>,
    /// Build statistics from trust-codegen compilation (logged at startup).
    ///
    /// Part of #4118.
    pub(super) trust_cg_build_stats: Option<super::trust_cg_dispatch::TrustCgBuildStats>,
    /// Frontend-neutral setup/execution timing evidence for the current checker run.
    ///
    /// TLA's trust-codegen path populates this from existing setup and compiled-BFS
    /// timing boundaries without changing checker behavior.
    pub(super) setup_trace: Option<std::cell::RefCell<tla_mc_core::SetupTrace>>,
    /// Runtime dispatch statistics for trust-codegen per-action evaluation.
    ///
    /// Part of #4374: lets parity tests distinguish compiled coverage from
    /// native actions that actually executed without falling back.
    pub(super) trust_cg_action_dispatch_stats: super::trust_cg_dispatch::TrustCgActionDispatchStats,

    /// Hybrid per-action flat-view dispatch state (ty-side M0 of wishlist item
    /// 4). Lazily initialized on the first per-action successor generation;
    /// inert (byte-identical to prior behavior) unless `TY_HYBRID_FLAT_VIEW` is
    /// set. See [`super::hybrid_dispatch`].
    pub(super) hybrid_dispatch: super::hybrid_dispatch::HybridDispatchState,
    /// Per-action native cache compiled against the HYBRID flat-view layout
    /// (item 4 M0-G1). Entirely separate from `trust_cg_cache` (whole-state
    /// layout): the two are compiled against different buffer geometries and
    /// carry disjoint artifact/warm-cache identities, so they can never cross.
    /// Only built under `TY_HYBRID_FLAT_VIEW=1` + `TY_HYBRID_NATIVE=1` on a
    /// compound (not fully flat) spec with >=1 flat-admissible variable.
    pub(super) trust_cg_hybrid_cache: Option<super::trust_cg_dispatch::TrustCgNativeCache>,
    /// The hybrid jit layout `trust_cg_hybrid_cache` was compiled against
    /// (`check_layout_to_jit_layout(hybrid check layout).with_hybrid_flat_view()`),
    /// kept for width/offset parity assertions at dispatch time.
    pub(super) trust_cg_hybrid_jit_layout: Option<tla_jit_abi::StateLayout>,
    /// Reusable output scratch for hybrid native action evaluation (mirrors
    /// `jit_action_out_scratch`, but sized to the hybrid compact slot width).
    pub(super) hybrid_action_out_scratch: Vec<i64>,

    // ==================== Composed sub-structs (Part of #1268) ====================
    /// Checkpoint state for periodic saves during model checking
    pub(super) checkpoint: CheckpointState,
    /// Partial Order Reduction state
    pub(super) por: PorState,
    /// Periodic liveness checking state (TLC doPeriodicWork pattern, Part of #2752)
    pub(super) periodic_liveness: PeriodicLivenessState,
    /// Debug instrumentation (only active in debug builds)
    pub(super) debug: DebugDiagnostics,
    /// Portfolio racing verdict for early exit when another lane resolves.
    /// Part of #3717: when `Some`, the BFS worker loop checks periodically
    /// and publishes its verdict upon completion.
    pub(crate) portfolio_verdict: Option<Arc<crate::shared_verdict::SharedVerdict>>,
    /// Cooperative state for fused BFS+symbolic mode (CDEMC).
    /// Part of #3767, Epic #3762.
    #[cfg(feature = "ay")]
    pub(crate) cooperative: Option<Arc<crate::cooperative_state::SharedCooperativeState>>,
    /// Deferred `TY_RP_VALUE=1` non-atomic refcount engagement (fused mode).
    ///
    /// Armed by `run_bfs_loop` when the opt-in is set and this sequential
    /// checker is the fused orchestrator's BFS lane (`cooperative` is `Some`):
    /// the exploration must START in atomic mode (the symbolic lanes are
    /// live), but may flip to the non-atomic fast path the moment every
    /// auxiliary orchestrator thread has fully terminated
    /// (`SharedCooperativeState::aux_lanes_terminated`). Polled at a low duty
    /// cycle from the sequential transport; cleared on flip or loop exit.
    #[cfg(feature = "ay")]
    pub(in crate::check::model_checker) rp_deferred_nonatomic_armed: bool,
    /// Poll-rate limiter for the deferred Rp engagement check (only every
    /// 64th dequeued state pays the handle inspection while armed).
    #[cfg(feature = "ay")]
    pub(in crate::check::model_checker) rp_deferred_poll_tick: u32,
    /// Collision detection for fingerprint-based state storage.
    pub(super) collision_detector: Option<crate::collision_detection::CollisionDetector>,

    /// Flat i64 state layout inferred from the first initial state.
    ///
    /// Populated by `infer_flat_state_layout()` after init state solving.
    /// Maps each state variable to a contiguous region of i64 slots in a
    /// flat buffer. Used by JIT-compiled transition functions and invariant
    /// checkers to operate on `FlatState` representations directly.
    ///
    /// `None` before init states are computed or when inference is skipped
    /// (e.g., no initial states generated).
    ///
    /// Part of #3986: Wire FlatState into BFS path.
    pub(super) flat_state_layout: Option<Arc<crate::state::StateLayout>>,

    /// Bridge for converting between `ArrayState` and `FlatState` at the
    /// BFS engine boundary.
    ///
    /// Created after `infer_flat_state_layout()` completes. Provides cheap
    /// `ArrayState <-> FlatState` conversions and fingerprint bridging.
    ///
    /// Part of #3986: Wire FlatState into BFS engine.
    pub(super) flat_bfs_bridge: Option<crate::state::FlatBfsBridge>,

    /// BFS-specific adapter wrapping the `FlatBfsBridge` with convenience
    /// methods for the interpreter sandwich: FlatState -> ArrayState -> eval ->
    /// ArrayState -> FlatState.
    ///
    /// Created alongside `flat_bfs_bridge` during layout inference. Tracks
    /// per-run conversion statistics. Auto-activated for fully-flat layouts
    /// (all scalar vars), or force-enabled via `use_flat_state=Some(true)`.
    /// See `should_use_flat_bfs()` for the full decision hierarchy.
    ///
    /// Part of #4126: FlatState as native BFS representation (Phase E).
    pub(super) flat_bfs_adapter: Option<crate::state::FlatBfsAdapter>,

    /// When true, all spec variables are scalar (Int/Bool) and `FlatState` is
    /// the primary BFS representation. Eliminates flatten/unflatten overhead
    /// because states are stored natively as `[i64]` buffers.
    ///
    /// Set during layout inference in `infer_flat_state_layout` when the layout
    /// `is_all_scalar()` returns true, roundtrip is verified, and no VIEW/SYMMETRY
    /// is active. When this is true, the BFS hot path can skip the interpreter
    /// sandwich (FlatState -> ArrayState -> eval -> ArrayState -> FlatState) and
    /// instead pass `&[i64]` directly to JIT-compiled transition functions.
    ///
    /// Part of #3986: Flat i64 state as primary BFS representation.
    pub(super) flat_state_primary: bool,
    /// Homotopic Canonicalizer for geometric symmetry reduction.
    /// (Part of Geometric Supremacy program).
    pub(super) homotopic_canonicalizer:
        Option<super::bfs::topology::canonicalize::HomotopicCanonicalizer>,
    /// WP-11 slice 2 (wishlist item 9): the verified flat-space symmetry
    /// canonicalizer, compiled per-layout from the declared SYMMETRY group.
    ///
    /// Installed only by `maybe_install_flat_symmetry_canonicalizer` (gated
    /// behind `TY_FLAT_SYMMETRY=1`, default OFF) after fail-closed admission:
    /// declared symmetry present, safety-only run, fully-flat
    /// roundtrip-verified layout, and `FlatSymmetryCanonicalizer::compile`
    /// succeeding for the FULL layout. When present (and the admission
    /// conditions still hold), the BFS fingerprint domain becomes
    /// `BfsFingerprintDomain::FlatSymmetryCanonical` and the flat-buffer
    /// canonicalization authority becomes
    /// `FlatBufferCanonicalizationAuthority::FlatSymmetry` — mutually
    /// exclusive with the legacy homotopic hook by construction of the
    /// authority derivation (see `model_checker/fingerprint.rs`).
    pub(super) flat_symmetry_canonicalizer:
        Option<std::sync::Arc<crate::state::flat_symmetry::FlatSymmetryCanonicalizer>>,
    /// FROZEN per-variable nested-set layouts + per-successor escape monitors
    /// (nested-set discovery A5 — the SOUNDNESS GATE).
    ///
    /// Populated by `freeze_nested_set_monitors` after the discovery prefix
    /// converges, when `NESTED_SET_PROMOTION_ENABLED` is set and the spec has a
    /// set-of-sets state variable (the `SlidingPuzzles` `board`). Each monitor
    /// guards one variable: on EVERY successor it checks membership against the
    /// frozen universe and FAILS CLOSED (bails the var to the interpreter's raw
    /// `value_fingerprint` for the rest of the run) on any out-of-universe board.
    /// The monitored dedup fingerprint byte-matches `value_fingerprint(board)`,
    /// so an in-universe board and a bailed board share ONE fingerprint domain
    /// (no aliasing, verdict identical). Empty for every spec without a
    /// set-of-sets variable, so non-nested specs are byte-identical and never
    /// pay for the pass.
    pub(super) nested_set_monitors: Vec<crate::state::NestedSetVarMonitor>,
    /// Native slide-kernel successor fast-path for the sliding-piece
    /// set-of-sets class (`SlidingPuzzles`) — Step B. `Some` when either
    ///
    /// * the STATIC RECOGNIZER ([`super::slide_recognize`]) PROVED at BFS
    ///   start that the spec's `Next` is the rigid-unit-slide relation
    ///   (the DEFAULT-ON path — killable with `TY_NO_NESTED_SET_SLIDE=1`), or
    /// * `TY_NESTED_SET_SLIDE=1` force-armed a nested-set board variable
    ///   (the original opt-in override, INIT-bounding-box grid).
    ///
    /// `None` for every other run, so the interpreter path is byte-identical.
    /// When armed, `generate_successors_array_raw` generates the variable's
    /// successors by word-ops over piece bitmasks instead of interpreting
    /// `Next`, failing closed (falling back to the interpreter) on any board
    /// that escapes the position grid.
    pub(super) nested_set_slide_arm: Option<crate::state::SlideKernelArm>,
    /// Always-on arm-time TRIPWIRE for the DEFAULT (recognizer-proven) slide
    /// arm: while non-zero, each kernel-generated successor set is compared
    /// against the interpreter's and the counter decremented; any divergence
    /// DISARMS the kernel (loudly) and the state — plus the rest of the run —
    /// falls back to the interpreter. Bounded (first
    /// [`super::run_bfs_full::SLIDE_TRIPWIRE_STATES`] states), so its cost is
    /// a run-constant. Zero for forced arms and unarmed runs.
    pub(super) nested_set_slide_tripwire: usize,
}

/// A captured snapshot of the explored state graph, for test assertions.
///
/// Only available with the `testing` feature.
#[cfg(feature = "testing")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateGraphSnapshot {
    /// Fingerprints of all captured nodes.
    pub nodes: Vec<u64>,
    /// Adjacency list: each node's fingerprint with its successor fingerprints.
    pub edges: Vec<(u64, Vec<u64>)>,
    /// Number of nodes that have at least one recorded parent.
    pub parent_count: usize,
    /// Total number of successor edges.
    pub successor_count: usize,
}

#[cfg(feature = "testing")]
impl<'a> ModelChecker<'a> {
    /// Test helper: forces the checker to capture the full state graph during checking.
    pub fn enable_state_graph_capture_for_testing(&mut self) {
        // Set the persistent testing override (not just `cache_for_liveness`):
        // `refresh_liveness_cache_requirement()` runs during check setup and
        // recomputes `cache_for_liveness` from the spec's liveness properties,
        // which would otherwise clobber a bare flag for specs with no PROPERTY.
        // The override is OR-ed back into `cache_for_liveness` on every refresh.
        self.liveness_cache.force_capture_for_testing = true;
        self.liveness_cache.cache_for_liveness = true;
    }

    /// Test helper: returns a [`StateGraphSnapshot`] of the captured state graph.
    #[must_use]
    pub fn state_graph_snapshot_for_testing(&self) -> StateGraphSnapshot {
        let mut nodes: Vec<u64> = self
            .liveness_cache
            .successors
            .collect_all_fingerprints()
            .into_iter()
            .map(|fp| fp.0)
            .collect();
        nodes.sort_unstable();
        nodes.dedup();

        let mut edges: Vec<(u64, Vec<u64>)> = if let Some(map) =
            self.liveness_cache.successors.as_inner_map()
        {
            map.iter()
                .map(|(&parent, successors)| {
                    let mut successors: Vec<u64> = successors.iter().map(|succ| succ.0).collect();
                    successors.sort_unstable();
                    (parent.0, successors)
                })
                .collect()
        } else {
            nodes
                .iter()
                .filter_map(|&parent| {
                    self.liveness_cache
                        .successors
                        .get(&Fingerprint(parent))
                        .map(|successors| {
                            let mut successors: Vec<u64> =
                                successors.into_iter().map(|succ| succ.0).collect();
                            successors.sort_unstable();
                            (parent, successors)
                        })
                })
                .collect()
        };
        edges.sort_unstable_by_key(|(parent, _)| *parent);
        let successor_count = edges.iter().map(|(_, successors)| successors.len()).sum();

        StateGraphSnapshot {
            nodes,
            parent_count: edges.len(),
            successor_count,
            edges,
        }
    }
}

impl<'a> ModelChecker<'a> {
    /// Test accessor: whether the nested-set slide kernel is armed (recognizer
    /// default-arm or `TY_NESTED_SET_SLIDE=1` force-arm). Reads the live arm,
    /// so a tripwire disarm shows up as `false`.
    #[must_use]
    pub fn nested_set_slide_armed_for_testing(&self) -> bool {
        self.nested_set_slide_arm.is_some()
    }

    /// Test accessor: `(actions_compiled, actions_total)` from the trust-cg build,
    /// or `None` if native codegen did not run.
    #[must_use]
    pub fn trust_cg_action_coverage_for_testing(&self) -> Option<(usize, usize)> {
        self.trust_cg_build_stats
            .as_ref()
            .map(|stats| (stats.actions_compiled, stats.actions_total()))
    }

    /// Number of actions recognized as the runtime-domain multi-successor
    /// ("NextStateLoop") shape but routed to the interpreter because the native
    /// multi-successor ABI codegen is not yet implemented. Used by the
    /// `#[ignore]`d `glowing_raccoon_next_state_loop` parity test to document
    /// the target for full native multi-successor execution.
    #[must_use]
    pub fn trust_cg_next_state_loop_recognized_for_testing(&self) -> Option<usize> {
        self.trust_cg_build_stats
            .as_ref()
            .map(|stats| stats.next_state_loop_recognized_unsupported)
    }

    /// Test accessor: trust-cg action-dispatch counts as
    /// `(enabled, disabled, runtime_errors)`, or `None` if native codegen did not run.
    #[must_use]
    pub fn trust_cg_action_dispatch_stats_for_testing(&self) -> Option<(usize, usize, usize)> {
        self.trust_cg_build_stats.as_ref()?;
        Some((
            self.trust_cg_action_dispatch_stats.enabled,
            self.trust_cg_action_dispatch_stats.disabled,
            self.trust_cg_action_dispatch_stats.runtime_errors,
        ))
    }

    /// Test accessor: the trust-cg native-admission evidence row, if recorded.
    #[must_use]
    pub fn trust_cg_native_admission_evidence_row_for_testing(&self) -> Option<&str> {
        self.trust_cg_build_stats
            .as_ref()
            .and_then(|stats| stats.native_admission_evidence_row.as_deref())
    }

    /// The trust-cg native-admission evidence report as a JSON value, if recorded.
    #[must_use]
    pub fn trust_cg_native_admission_evidence_report_json(&self) -> Option<serde_json::Value> {
        self.trust_cg_build_stats
            .as_ref()
            .and_then(|stats| stats.native_admission_evidence_report.as_ref())
            .map(|report| report.to_json_value())
    }

    /// Return a summarizer-ready backend capability report for JSON/JSONL sinks.
    ///
    /// The returned object is intended to be serialized under the
    /// `backend_capability_report` key, which is one of the wrapper keys
    /// consumed by the `ty-mcc-summarize-evidence` binary.
    #[must_use]
    pub fn backend_capability_report_json(&self) -> Option<serde_json::Value> {
        {
            self.trust_cg_native_admission_evidence_report_json()
        }
    }
}

#[cfg(test)]
impl<'a> ModelChecker<'a> {
    pub(in crate::check) fn test_vars(&self) -> &[Arc<str>] {
        &self.module.vars
    }

    pub(in crate::check) fn test_fairness(&self) -> &[FairnessConstraint] {
        &self.liveness_cache.fairness
    }

    pub(in crate::check) fn test_seen_is_empty(&self) -> bool {
        self.state_storage.seen.is_empty()
    }

    /// Test helper: whether the checker is in full-state storage mode.
    /// Used to assert the former SYMMETRY+liveness auto-upgrade stays gone
    /// (declared SYMMETRY is now ignored for genuine temporal properties).
    pub(in crate::check) fn test_store_full_states(&self) -> bool {
        self.state_storage.store_full_states
    }

    pub(in crate::check) fn test_seen_fps_len(&self) -> usize {
        self.state_storage.seen_fps.len()
    }

    /// Test helper: inject fp into seen_fps to create trace.depths mismatch.
    pub(in crate::check) fn test_inject_spurious_fingerprint(&self, fp: Fingerprint) {
        self.state_storage.seen_fps.insert_checked(fp);
    }
}
