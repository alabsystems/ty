// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! IC3 configuration types, parameter structs, and constants.

use crate::sat_types::SolverBackend;

/// Validation strategy for IC3 invariant checking (#4121).
///
/// Controls how aggressively the engine validates the inductive invariant
/// after convergence. High constraint-to-latch ratio circuits can overwhelm
/// even AYNoPreprocess during the Inv AND T => Inv' check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValidationStrategy {
    /// Full 3-tier validation. Default. Uses constraint/latch ratio for tier selection.
    #[default]
    Auto,
    /// Skip the expensive per-lemma Inv AND T => Inv' check entirely.
    /// Only performs Init => Inv and Inv => !Bad checks.
    #[deprecated(note = "unsound in portfolio mode — use Auto. See Wave 26 P0 soundness bug.")]
    SkipConsecution,
    /// Skip all validation. ONLY safe when another portfolio member validates.
    None,
}

/// Default CTG parameters for MIC generalization.
/// CTG_MAX: max CTG attempts per literal drop in MIC.
/// CTG_LIMIT: shared blocking-goal budget for one `block_ctg_chain` call.
pub(super) const DEFAULT_CTG_MAX: usize = 3;
pub(super) const DEFAULT_CTG_LIMIT: usize = 1;

/// Default CTP parameters for propagation.
/// CTP_MAX: max CTP attempts per lemma push in propagate().
///
/// ty default: 2 — the common failure is a single spurious predecessor
/// blocking the push, which the first retry clears; each further attempt
/// costs another propagation SAT call plus a CTG blocking chain per lemma
/// per frame for diminishing convergence gains. Circuit-tuned configs
/// raise it where retries pay (e.g. arithmetic portfolio configs use
/// `ctp_max: 5`).
pub(super) const DEFAULT_CTP_MAX: usize = 2;

/// Base threshold for periodic solver rebuild. The actual threshold is
/// scaled by circuit size: `base * max(1, num_latches / 20)`. Small circuits
/// (< 20 latches) rebuild at the base rate; larger circuits get proportionally
/// more headroom before rebuilding, since each MIC iteration is more expensive
/// and rebuilding all solvers on a 200-latch circuit with 50+ frames is costly.
///
/// The old fixed 10K threshold was too aggressive for large circuits (causing
/// frequent expensive rebuilds) and too lenient for tiny circuits.
pub(super) const SOLVER_REBUILD_BASE: usize = 5_000;

/// Default sampling interval for independent consecution verification (#4092).
///
/// The actual interval is computed adaptively by `consecution_verify_interval()`
/// based on the circuit's clause-to-latch ratio. This constant is the default
/// for low-ratio circuits where ay-sat false UNSAT is rare.
///
/// See `consecution_verify_interval()` for the adaptive logic.
pub(super) const CONSECUTION_VERIFY_INTERVAL_DEFAULT: usize = 10;

/// Upper bound on latch count where `verify_consecution_independent`
/// remains cheap enough to run as a MIC soundness cross-check (#4092).
///
/// Above this threshold, SimpleSolver becomes too slow to check every MIC
/// generalization, so the engine trusts ay-sat's result and relies on
/// post-convergence validation for the broader soundness net.
pub(super) const VERIFY_CONSECUTION_INDEPENDENT_MAX_LATCHES: usize = 60;

/// Compute the adaptive consecution verification interval (#4121).
///
/// High constraint-to-latch ratio circuits cause ay-sat to produce false UNSAT
/// more frequently because the SAT solver spends disproportionate time on
/// constraint propagation and may terminate prematurely. Sampling-based
/// verification (checking only every Nth consecution) misses too many unsound
/// lemmas on these circuits.
///
/// Two metrics are considered:
/// 1. `trans_clauses / latches` — Tseitin encoding complexity. A circuit can have
///    0 explicit AIGER constraints but thousands of trans_clauses that stress ay-sat.
///    cal76 (36 consecution errors) has few constraint_lits but many trans_clauses.
/// 2. `constraint_lits / latches` — AIGER environment constraint density. Circuits
///    like microban (100-300+ constraint_lits on 20-60 latches) overwhelm the SAT
///    solver's constraint propagation path even when trans_clauses are moderate.
///    microban_148 (false UNSAT) and microban_1 (infinite rebuild) are in this class.
///
/// The effective ratio is the MAX of both metrics, since either high trans_clauses
/// OR high constraint_lits can trigger ay-sat false UNSAT.
///
/// Thresholds:
/// - ratio > 5.0: verify every consecution (interval=1). These circuits produce
///   false UNSAT frequently enough that sampling is dangerous.
/// - ratio > 2.0: verify every 3rd (interval=3). Moderate risk.
/// - ratio <= 2.0: verify every 10th (interval=10). Low risk, save overhead.
#[allow(dead_code)]
pub(super) fn consecution_verify_interval(num_trans_clauses: usize, num_latches: usize) -> usize {
    consecution_verify_interval_full(num_trans_clauses, 0, num_latches)
}

/// Compute the adaptive consecution verification interval considering both
/// trans_clauses and constraint_lits (#4121).
///
/// This is the full version that takes both metrics. The shorter
/// `consecution_verify_interval()` is kept for backward compatibility.
pub(super) fn consecution_verify_interval_full(
    num_trans_clauses: usize,
    num_constraints: usize,
    num_latches: usize,
) -> usize {
    if num_latches == 0 {
        return 1;
    }
    // Small-circuit fast path (#4259, #4288): skip cross-check entirely on
    // circuits with fewer than 30 latches. On cal14 (23 latches, 1656 trans
    // clauses, 72x ratio) the current adaptive logic sets interval=1 which
    // verifies every consecution — but SimpleSolver's basic DPLL is unreliable
    // on clause-dense small circuits and produces false SAT. That causes
    // ay-sat's correct UNSAT lemmas to be rejected indefinitely: IC3 stays at
    // depth=1 with 0 lemmas learned and times out.
    //
    // Returning usize::MAX here effectively disables the sampling-based
    // cross-check (block.rs:343 divides depth by the interval — any division
    // that yields 0 skips the check). Post-convergence
    // `validate_invariant_budgeted()` still runs and provides the soundness
    // safety net.
    if num_latches < 30 {
        return usize::MAX;
    }
    let trans_ratio = num_trans_clauses as f64 / num_latches as f64;
    let constraint_ratio = num_constraints as f64 / num_latches as f64;
    // Use the max of both metrics: either high trans_clauses OR high
    // constraint_lits can cause ay-sat false UNSAT.
    let ratio = trans_ratio.max(constraint_ratio);
    if ratio > 5.0 {
        1
    } else if ratio > 2.0 {
        3
    } else {
        CONSECUTION_VERIFY_INTERVAL_DEFAULT
    }
}

/// Determine if a circuit has a high constraint-to-latch ratio (#4121).
///
/// High-constraint circuits (microban class: 100-300+ constraints on 20-60
/// latches) cause systematic ay-sat false UNSAT and SimpleSolver false SAT
/// in cross-checks. For these circuits, the cross-check should be disabled
/// from the start rather than waiting for the failure budget to be exhausted,
/// because:
/// 1. SimpleSolver's basic DPLL produces false SAT on constraint-dense formulas
/// 2. ay-sat's incremental queries are unreliable on these formulas
/// 3. The only reliable soundness check is the post-convergence validation
///
/// Returns true if either:
/// - `constraint_lits / latches > 5` (AIGER constraint density)
/// - `trans_clauses / latches > 10` AND `constraint_lits > latches` (combined pressure)
pub(super) fn is_high_constraint_circuit(
    num_trans_clauses: usize,
    num_constraints: usize,
    num_latches: usize,
) -> bool {
    if num_latches == 0 {
        return false;
    }
    let constraint_ratio = num_constraints as f64 / num_latches as f64;
    let trans_ratio = num_trans_clauses as f64 / num_latches as f64;
    // Pure constraint density: microban class (124 constraints / 23 latches = 5.4x)
    constraint_ratio > 5.0
        // Combined pressure: high trans + non-trivial constraints
        || (trans_ratio > 10.0 && num_constraints > num_latches)
}

/// Number of consecutive SatResult::Unknown from non-poisoned solvers before
/// the engine falls back to AYNoPreprocess (#4074). ay-sat can produce
/// FINALIZE_SAT_FAIL on certain clause structures (e.g., cal14), causing
/// the solver to return Unknown/InvalidSatModel indefinitely. After
/// this many consecutive Unknown results, we disable preprocessing entirely.
///
/// Note: `solve_incremental_ic3()` skips `preprocess()`, so BVE is not the
/// cause of these failures. The fallback is a general resilience mechanism.
pub(super) const UNKNOWN_FALLBACK_THRESHOLD: usize = 3;

/// Maximum proof obligation depth. If an obligation chain exceeds this,
/// we stop exploring that branch and return to the queue. This prevents
/// runaway depth explosion when the transition system has very long
/// reachability chains. The BMC engine is better suited for such cases.
///
/// Set high enough that it doesn't interfere with genuine counterexamples
/// on HWMCC benchmarks (deepest known CEX is ~200 steps).
pub(super) const MAX_OBLIGATION_DEPTH: usize = 500;

/// Maximum cross-check failures per frame before disabling the independent
/// consecution verification for that frame. When ay-sat consistently produces
/// false UNSAT for a particular frame's clause structure (SimpleSolver
/// disagrees), rebuilding the solver and retrying creates an infinite loop.
/// After this many failures at the same frame, we stop cross-checking and
/// instead skip the ay-sat UNSAT result entirely (treat as inconclusive).
///
/// This breaks the rebuild loop observed on microban_1 (UNSAT) and similar
/// benchmarks where ay-sat returns false UNSAT 30K+ times in 10 seconds.
pub(super) const MAX_CROSSCHECK_FAILURES: usize = 5;

/// Maximum TOTAL cross-check failures across all frames (#4121).
///
/// Catches distributed failure patterns where ay-sat false UNSAT affects many
/// frames (each under the per-frame `MAX_CROSSCHECK_FAILURES` threshold).
/// Without this, a circuit that triggers failures at frame 1, then frame 2,
/// then frame 3, etc. would never hit the per-frame threshold but would still
/// suffer repeated expensive solver rebuilds.
///
/// Value = 2 * MAX_CROSSCHECK_FAILURES: slightly above per-frame max to allow
/// normal per-frame recovery before triggering, but low enough to catch
/// distributed patterns within 10 total failures.
pub(super) const MAX_TOTAL_CROSSCHECK_FAILURES: usize = 10;

/// Maximum times a proof obligation can be re-queued due to Unknown results
/// before being dropped. When the solver backend is already at its simplest
/// (SimpleSolver) and still returns Unknown, re-queuing the PO is futile.
/// This prevents the secondary infinite loop where Unknown POs cycle forever.
pub(super) const MAX_UNKNOWN_REQUEUES: usize = 3;

/// Maximum number of spurious init-consistent predecessors before skipping
/// verify_trace entirely for init-consistent predecessors (#4105).
pub(super) const MAX_SPURIOUS_INIT_PREDS: usize = 3;

/// Maximum number of solver rebuilds per frame before giving up on that
/// frame's solver and treating Unknown/poisoned results conservatively (#4105).
///
/// When a solver at frame `i` is rebuilt more than this many times (due to
/// poisoning from ay-sat panics or Unknown results triggering rebuilds),
/// further rebuilds are skipped and the query result is treated as Sat
/// (conservative: no false UNSAT). This prevents the infinite rebuild loop
/// observed on microban_1 where constraint-dense formulas cause repeated
/// solver corruption at the same frame.
pub(super) const MAX_SOLVER_REBUILDS_PER_FRAME: usize = 3;

/// Number of ay-sat panics before falling back to SimpleSolver (#4092).
///
/// ay-sat panics indicate internal corruption (backtrack bugs, conflict analysis
/// errors). The corruption may have produced incorrect UNSAT results *before*
/// the panic manifested — the panic is just the final symptom. After this many
/// panics, fall back to SimpleSolver and purge all frame lemmas.
///
/// Set to 1: a single panic means the solver is unreliable on this circuit.
/// For small circuits (<50 latches), SimpleSolver is fast enough.
pub(super) const PANIC_FALLBACK_THRESHOLD: usize = 1;

/// Maximum times `get_bad()` may return the same bad cube at a given depth
/// before the blocking loop advances to the next depth (#4139).
///
/// Several early-out paths can leave a bad cube unblocked without recording
/// a counterexample — the obligation-depth cap (`MAX_OBLIGATION_DEPTH`), the
/// Unknown-requeue cap (`MAX_UNKNOWN_REQUEUES`), and the spurious-init-pred
/// loop breaker (#4105). When that happens, `get_bad()` rediscovers the same
/// bad state on the next query, and without a cap the blocking loop would
/// cycle on it forever.
///
/// After `MAX_BAD_CUBE_REPEATS` rediscoveries of the same cube, the blocking
/// loop breaks and IC3 advances to the next depth. This is sound: IC3 frames
/// are over-approximations, and the unblocked cube will be re-examined at the
/// next depth where additional frame lemmas may enable successful blocking.
pub(super) const MAX_BAD_CUBE_REPEATS: usize = 10;

/// Literal ordering strategy for MIC generalization.
///
/// Controls the order in which literals are tried for removal during
/// the MIC (Minimal Inductive Clause) generalization loop. Different
/// orderings lead to different generalization paths and complementary
/// coverage across portfolio members.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GeneralizationOrder {
    /// Sort by VSIDS activity (default). Low-activity literals are tried
    /// for removal first (backward iteration over activity-sorted array).
    #[default]
    Activity,
    /// Reverse topological order: drop output-side latches before input-side.
    /// Uses the AND-gate depth from each latch to the primary inputs.
    /// Latches further from inputs (deeper in the circuit) are tried for
    /// removal first, since they depend on more variables and are more
    /// likely to be don't-cares in the generalization.
    ReverseTopological,
    /// Random shuffle using the config's random_seed. Different seeds
    /// produce different orderings, providing pure diversity without any
    /// heuristic bias. Useful in portfolios where activity-based ordering
    /// might consistently miss certain generalizations.
    RandomShuffle,
}

/// Restart strategy hint for IC3 frame solvers.
///
/// These are advisory hints stored in the IC3 configuration. Currently
/// ay-sat manages its own restart schedule internally, so these are not
/// directly applied. They serve as portfolio diversity knobs for future
/// integration when ay-sat supports configurable restart strategies.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RestartStrategy {
    /// Use the solver's default restart strategy.
    #[default]
    Default,
    /// Geometric restart: conflicts_to_restart = base * factor^restarts.
    /// More aggressive restarts for smaller base values.
    Geometric {
        /// Conflict count before the first restart.
        base: usize,
        /// Multiplier applied to the interval after each restart.
        factor: f64,
    },
    /// Luby sequence restart: conflicts_to_restart = unit * luby(restarts).
    /// Provides a good balance of short and long restart intervals.
    Luby {
        /// Base unit scaled by the Luby sequence term for each restart.
        unit: usize,
    },
}

/// IC3 engine configuration for portfolio diversity.
///
/// Each field controls a knob that can be varied across portfolio members.
/// The published rIC3 system description (arXiv:2502.13605 §4) fields a
/// 16-engine pool in which 11 slots are differing IC3 variants; the
/// configuration axes below are ty's own surface for producing that
/// diversity.
#[derive(Debug, Clone)]
pub struct Ic3Config {
    /// Enable CTP (Counter-To-Propagation) in propagate().
    pub ctp: bool,
    /// Enable infinity frame promotion.
    pub inf_frame: bool,
    /// Enable internal signals (FMCAD'21 AND gate variables in cubes).
    pub internal_signals: bool,
    /// Enable "inn-proper": promote internal signals to first-class latches
    /// (FMCAD'21 at the state-variable basis). Mutually exclusive with
    /// `internal_signals` (which is the cube-extension variant).
    ///
    /// When enabled, AND-gate outputs that do not depend on primary inputs
    /// are promoted to latches with next-state functions derived from a 1-step
    /// unroll. IC3 frames become clauses over `latches ∪ promoted_signals`,
    /// yielding structurally smaller inductive invariants on arithmetic-heavy
    /// circuits where latch-only lemmas are too long to express the invariant.
    ///
    /// Reference: internal-signal latch promotion (FMCAD'21).
    /// Issue: #4308.
    pub inn_proper: bool,
    /// Enable ternary simulation pre-reduction in find_bad_in_frame.
    pub ternary_reduce: bool,
    /// Max CTG (Counter-To-Generalization) attempts per literal drop in MIC.
    /// Higher values = more generalization effort. Default: 3.
    /// ty's portfolio varies this axis over 0..=8 (0 = CTG off, 8 = the
    /// Sokoban-tuned deep variant, #4284).
    pub ctg_max: usize,
    /// Shared blocking-goal budget for one `block_ctg_chain` call during CTG.
    /// Higher values allow deeper predecessor blocking. Default: 1.
    /// ty's portfolio varies this axis over 0..=12. The raised value 12 is
    /// ty's static analogue of the published dynamic EXCTG budget
    /// (arXiv:2501.02480 Alg. 5): that formula opens at ~5 when extended
    /// CTG activates and reaches 12 at ~2.5x the activation activity,
    /// growing only sub-linearly beyond — so a fixed budget of 12 captures
    /// most of the band's headroom.
    pub ctg_limit: usize,
    /// Enable circuit-size-based CTG adaptation at engine construction time.
    ///
    /// When enabled, adjusts CTG parameters based on circuit size:
    /// - Very small circuits (<30 latches): conservative CTG
    ///   (ctg_max=min(cfg,3), ctg_limit=min(cfg,1))
    /// - Small-medium circuits (30..100 latches): aggressive CTG
    ///   (ctg_max=max(cfg,5), ctg_limit=max(cfg,12))
    /// - Medium circuits (100..=500 latches): use configured values as-is
    /// - Large circuits (>500 latches): conservative CTG (ctg_max=min(cfg,2), ctg_limit=min(cfg,1))
    pub circuit_adapt: bool,
    /// Max CTP attempts per lemma push in propagate().
    /// Only used when `ctp` is true. Default: 2 (`DEFAULT_CTP_MAX`).
    pub ctp_max: usize,
    /// Random seed for activity initialization and randomized MIC ordering,
    /// providing tie-breaking diversity across portfolio members.
    /// Default: 0 (no perturbation).
    pub random_seed: u64,
    /// Literal ordering strategy for MIC generalization.
    pub gen_order: GeneralizationOrder,
    /// SAT solver backend. Default: AYSat (the production default).
    /// See [`SolverBackend`] for available options and trade-offs.
    pub solver_backend: SolverBackend,
    /// Enable consecutive-Unknown fallback escalation (#4233).
    ///
    /// When true, persistent non-poisoned `SatResult::Unknown` answers trigger
    /// the historical FINALIZE_SAT_FAIL defense path:
    /// `AYSat -> AYNoPreprocess -> SimpleSolver`.
    ///
    /// Keep enabled by default for production resilience. Disable only for
    /// controlled sweep evidence that needs to measure the pinned ay-sat
    /// behavior without this fallback masking regressions.
    pub unknown_fallback_enabled: bool,
    /// Advisory restart strategy hint for IC3 frame solvers.
    pub restart_strategy: RestartStrategy,
    /// Enable parent lemma heuristic in MIC generalization.
    ///
    /// When generalizing a cube at frame k, find a "parent lemma" in frame k-1
    /// that subsumes the cube, and sort MIC literals so that parent-lemma
    /// literals are tried for removal last. This biases generalization toward
    /// reusing structure from already-proven lemmas.
    ///
    /// Reference: the CAV'23 parent-lemma heuristic.
    pub parent_lemma: bool,
    /// Enable CTG-enhanced down() MIC variant.
    ///
    /// When a literal-drop fails in MIC, instead of just marking the literal
    /// as essential, extract the counterexample model and shrink the cube by
    /// keeping only literals present in the model. This is more aggressive than
    /// standard literal-dropping and can find better generalizations.
    ///
    /// Reference: arXiv:2501.02480 Alg. 3 (`ctg_down`).
    pub ctg_down: bool,
    /// Enable dynamic generalization parameters (GAP-5).
    ///
    /// Instead of using fixed CTG parameters for all proof obligations,
    /// dynamically adjust them from the successor obligation's activity
    /// (see `dynamic_ctg_params`). High activity (indicating thrashing)
    /// gets more aggressive generalization:
    /// - activity < 10: no CTG (ctg_max=0, ctg_limit=0)
    /// - activity in [10, 40): plain CTG (ctg_max=(act-10)/10+2, ctg_limit=1)
    /// - activity >= 40: extended CTG (ctg_max=5, ctg_limit=(act-40)^0.3*2+5)
    ///
    /// This allows IC3 to invest more effort in cubes that are proving
    /// difficult to block, while keeping overhead low for easy cubes.
    ///
    /// Reference: arXiv:2501.02480 §IV, Alg. 5 (dynamic adjustment of
    /// generalization strategies; parameter formulas as published there,
    /// threshold values from the paper's §V-A experiment setup).
    pub dynamic: bool,
    /// Enable predicate propagation (backward analysis) for bad-state discovery.
    ///
    /// When enabled, IC3 uses a backward transition solver to find predecessors
    /// of bad states, complementing the standard forward `get_bad()` query.
    /// The backward solver encodes: Trans(s,s') AND bad(s') AND !bad(s).
    ///
    /// Predprop helps on benchmarks where the property has small backward
    /// reachability even if forward IC3 struggles with coarse frame approximations.
    pub predprop: bool,
    /// VSIDS decay factor for the bucket-queue VSIDS used in MIC literal ordering.
    ///
    /// Controls how quickly old activity scores fade relative to new bumps.
    /// Higher values (closer to 1.0) keep older bumps relevant longer, producing
    /// more stable literal orderings. Lower values (closer to 0.5) make the
    /// ordering more responsive to recent queries.
    ///
    /// Portfolio diversity: varying decay across configs produces different
    /// MIC literal orderings even with the same seed. Default: 0.99.
    /// Typical range: 0.75..0.999.
    pub vsids_decay: f64,
    /// Number of IC3 SAT restarts before switching from bucket-queue VSIDS
    /// (O(1) amortized variable selection) to binary-heap VSIDS (O(log n) exact).
    ///
    /// ty engineering choice: the bucket queue wins on the short queries
    /// that dominate early runs; once queries grow long enough to trigger
    /// restarts repeatedly, exact heap ordering pays for its O(log n). The
    /// switch is one-way and both modes are correct, so only the order of
    /// magnitude matters (see `DEFAULT_SWITCH_TO_HEAP_AFTER_RESTARTS` in
    /// `vsids.rs`).
    ///
    /// - `0` = start directly in heap mode (skip bucket queue entirely)
    /// - `8` = default ("first handful of restarts", untuned round value)
    /// - `50` = keep bucket queue longer (good for circuits with many short queries)
    pub bucket_queue_restarts: usize,
    /// Maximum number of literal-drop SAT calls per MIC invocation (#4072).
    ///
    /// Limits the expensive per-literal inductiveness checks in the MIC
    /// drop loop. For arithmetic circuits where carry chains make most
    /// literals essential, this prevents O(n) wasted SAT calls.
    ///
    /// - `0` = unlimited (default, standard MIC behavior)
    /// - `N > 0` = stop after N failed+successful drops combined
    pub mic_drop_budget: usize,
    /// Enable model-unassignment don't-care detection in the lift/MIC paths (#4091).
    ///
    /// After a SAT result, `SatSolver::unassign_model_value` asks the solver
    /// whether a variable's model assignment can be retracted without leaving
    /// any clause unsatisfied — a zero-SAT-call don't-care test backed by
    /// ay-sat's `flip_to_none` primitive. When this flag is set, MIC CTG-down
    /// uses it to shrink candidate cubes literal-by-literal against the
    /// current model instead of paying one SAT call per literal.
    ///
    /// Enabled by default for ay-sat backends (which implement the primitive).
    /// Falls back gracefully on backends where `unassign_model_value` returns
    /// false (the literal is simply kept).
    pub flip_to_none_lift: bool,
    /// Maximum number of `get_bad()` SAT calls per depth level (#4072).
    ///
    /// On arithmetic circuits with 64+ latches, the blocking phase at each
    /// depth level may never terminate because the number of distinct bad cubes
    /// is exponential. The blocking budget forces IC3 to advance to the next
    /// depth after N bad-state discoveries, even if `get_bad()` still finds
    /// reachable bad states.
    ///
    /// Advancing early is sound: IC3's frame sequence still over-approximates
    /// reachability. The unblocked cubes will be re-discovered at the next
    /// depth where they may be easier to block (more frame lemmas available).
    ///
    /// - `0` = unlimited (default, standard IC3 behavior)
    /// - `N > 0` = advance after N get_bad() SAT calls return bad cubes
    pub blocking_budget: usize,
    /// Number of variable orderings to try during MIC generalization (#4099).
    ///
    /// Multi-ordering lift runs MIC with multiple literal orderings and keeps
    /// the shortest (most general) result. Values:
    /// - `0` or `1` = disabled (standard single-ordering MIC)
    /// - `2` = try primary ordering + one complementary ordering
    /// - `3` = try primary + complementary + random shuffle (maximum diversity)
    ///
    /// Additional passes only attempted when first result > half original cube
    /// and circuit has > 15 latches.
    pub multi_lift_orderings: usize,
    /// Validation strategy for the post-convergence invariant check (#4121).
    ///
    /// Controls which validation checks are performed after IC3 converges.
    /// High-constraint-ratio circuits (e.g., qspiflash with 157 latches, ~800
    /// constraints) can timeout during validation even with AYNoPreprocess.
    /// Speed-focused portfolio configs can skip validation since at least one
    /// portfolio member (with Auto) always validates.
    pub validation_strategy: ValidationStrategy,
    /// Enable parent lemma MIC seeding optimization (CAV'23 #4150).
    ///
    /// When a proof obligation has a parent (po.next), and the parent's cube
    /// was previously blocked with a lemma, use the intersection of the
    /// current cube and the parent's blocking lemma as the starting point for
    /// MIC generalization. This produces tighter lemmas faster by reducing the
    /// initial literal set to structurally relevant variables.
    pub parent_lemma_mic: bool,
    /// Maximum consecutive failed literal drops before MIC aborts (#4244).
    ///
    /// When MIC tries to remove literals and N consecutive attempts fail (the
    /// literal is essential), assume the cube is approximately minimal and stop
    /// trying. This dramatically improves mics/second on circuits where most
    /// literals are essential (e.g., arithmetic carry chains).
    ///
    /// Reference: IC3ref `IC3.cpp:598-623` — `micAttempts` parameter (default 3).
    /// Bradley notes: "Definitely improves mics/second to use a low micAttempts,
    /// but does it improve overall performance?" (yes, on hard industrial circuits).
    ///
    /// - `0` = unlimited (never abort early, standard MIC behavior)
    /// - `N > 0` = abort after N consecutive failed drops. Reset counter on success.
    ///
    /// Default: 3 (matching IC3ref). Portfolio diversity: vary between 2..5.
    pub mic_attempts: usize,
    /// Disable consecution cross-checking from the start (#4163).
    ///
    /// When true, the independent consecution verification (SimpleSolver
    /// cross-check of ay-sat UNSAT results) is permanently disabled for this
    /// engine configuration. This is useful for:
    ///
    /// 1. **SimpleSolver backend configs**: SimpleSolver doesn't produce false
    ///    UNSAT, so cross-checking it against itself is pointless overhead.
    /// 2. **Speed-focused configs**: cross-checking adds overhead per consecution.
    ///    Configs that prioritize speed can skip it, relying on other portfolio
    ///    members (with crosscheck enabled) for soundness verification.
    /// 3. **High-constraint-ratio circuits**: these are also auto-detected at
    ///    engine construction time via `is_high_constraint_circuit()`, but
    ///    setting this in config makes the intent explicit.
    ///
    /// The post-convergence `validate_invariant_budgeted()` provides the
    /// ultimate soundness safety net regardless of this setting.
    ///
    /// Note: even when this is `false`, the engine may still auto-disable
    /// crosscheck at construction time for high-constraint circuits or at
    /// runtime when the cross-check failure budget is exhausted.
    pub crosscheck_disabled: bool,
    /// Enable inductive-subclause generalization during `push_lemma` (#4244).
    ///
    /// When pushing a lemma from frame `f_src` forward to a higher frame `f`,
    /// after relative inductiveness at `f` is confirmed, attempt to drop
    /// individual literals (in ascending VSIDS-activity order, low-activity
    /// first) and re-verify inductiveness. Each successful drop yields a
    /// strictly stronger lemma at `f` without additional frame propagation.
    ///
    /// This is a standard IC3 optimization (cf. Bradley's IC3ref
    /// `pushForward`). Disabled by default because push-time generalization
    /// can interact with domain restriction on small circuits in ways that
    /// cause premature convergence failures (the salvaged implementation
    /// reverted in #4244 caused 17 tests to return `Unknown`).
    ///
    /// When enabled, generalization is bounded by `push_generalize_budget`.
    pub push_generalize: bool,
    /// Maximum number of successful literal drops per push-time generalization.
    ///
    /// Only used when `push_generalize` is `true`. Caps the work spent per
    /// call so push_lemma stays O(|cube|) SAT calls in the worst case.
    /// Default: 4 — ty cap: push-time generalization is a bonus on top of
    /// the full MIC the lemma already received, so a handful of extra drop
    /// attempts captures the frame-local slack without turning every push
    /// into a second generalization pass.
    pub push_generalize_budget: usize,
    /// Disable domain-restricted SAT for small circuits (#4259, ay#8802 workaround).
    ///
    /// When true, IC3 skips all `set_domain()` calls on both the clause-filtered
    /// mini-solver and the full frame solvers. With no active domain, ay-sat
    /// falls back to `search_propagate_standard` (plain BCP) instead of
    /// `search_propagate_domain`. On circuits with fewer than ~50 latches the
    /// domain-BCP path adds per-query overhead that dominates the tiny solver
    /// cost, causing IC3 to stall on Tier 1 HWMCC benchmarks
    /// (cal14/cal42/loopv3/microban_1_UNSAT) that should solve in well
    /// under a second.
    ///
    /// This is a soundness-preserving workaround: domain restriction is a
    /// performance optimization, not a correctness property. Disabling it
    /// only affects how long each SAT call takes, not which clauses it sees
    /// (all clauses are still added to the solver; only the BCP watcher
    /// filtering is bypassed).
    ///
    /// The real fix for the ay-sat domain-BCP overhead is tracked in ay#8802
    /// (size threshold for `active_domain` activation). Until that lands,
    /// this flag provides a dependency-free tla-aiger workaround.
    ///
    /// Default: false. `Ic3Engine::with_config()` auto-enables this on circuits
    /// with fewer than 50 latches when not already set.
    pub small_circuit_mode: bool,
}

impl Default for Ic3Config {
    fn default() -> Self {
        // Conservative defaults: parent_lemma enabled (cheap, effective),
        // ctg_down disabled (more aggressive, may not always help).
        // dynamic disabled by default: activity-adaptive generalization is
        // a portfolio-diversity axis, not a universal win.
        Ic3Config {
            ctp: false,
            inf_frame: false,
            internal_signals: false,
            inn_proper: false,
            ternary_reduce: false,
            ctg_max: DEFAULT_CTG_MAX,
            ctg_limit: DEFAULT_CTG_LIMIT,
            circuit_adapt: false,
            ctp_max: DEFAULT_CTP_MAX,
            random_seed: 0,
            gen_order: GeneralizationOrder::Activity,
            solver_backend: SolverBackend::default(),
            unknown_fallback_enabled: true,
            restart_strategy: RestartStrategy::Default,
            parent_lemma: true,
            ctg_down: false,
            dynamic: false,
            predprop: false,
            vsids_decay: 0.99,
            bucket_queue_restarts: 8,
            flip_to_none_lift: true,
            mic_drop_budget: 0,
            blocking_budget: 0,
            multi_lift_orderings: 3,
            validation_strategy: ValidationStrategy::Auto,
            parent_lemma_mic: true,
            mic_attempts: 3,
            crosscheck_disabled: false,
            push_generalize: false,
            push_generalize_budget: 4,
            small_circuit_mode: false,
        }
    }
}

/// Result of an IC3 model checking run.
#[derive(Debug)]
pub enum Ic3Result {
    /// Property holds: system is safe. Contains the convergence depth and
    /// the inductive invariant (as CNF lemmas) for portfolio-level validation.
    Safe {
        /// Frame index at which the invariant converged (proof depth).
        depth: usize,
        /// Inductive invariant: conjunction of CNF clauses (lemmas) that proves safety.
        /// Each inner `Vec<Lit>` is a clause (disjunction of literals).
        /// Empty when the result comes from a degenerate case (no bad lits).
        lemmas: Vec<Vec<Lit>>,
    },
    /// Property violated: counterexample found.
    Unsafe {
        /// Depth of the counterexample trace.
        depth: usize,
        /// Counterexample trace: sequence of states from init to bad.
        /// Each state is a vector of (variable, value) pairs.
        trace: Vec<Vec<(Var, bool)>>,
    },
    /// Could not determine within resource limits.
    Unknown {
        /// Why the run was inconclusive (e.g. timeout, depth limit, cancellation).
        reason: String,
    },
}

/// Result of `get_bad()` — distinguishes standard bad states from
/// predicate-propagation predecessors so the main loop can route them
/// to different proof-obligation frames.
pub(super) enum GetBadResult {
    /// A state in F_k that satisfies the bad property (standard forward check).
    Bad(Vec<Lit>),
    /// A predecessor of bad found by backward analysis (predprop).
    /// This state is NOT bad itself — it's one transition step away from bad.
    Predecessor(Vec<Lit>),
}

/// Check if proof verification is explicitly enabled via environment variable (#4216).
///
/// Returns `true` when `TY_PROOF_VERIFY=1` is set, signaling that the
/// portfolio runner wants defense-in-depth validation with more generous
/// time budgets for larger circuits.
pub(super) fn proof_verification_enabled() -> bool {
    std::env::var("TY_PROOF_VERIFY")
        .map(|v| v == "1")
        .unwrap_or(false)
}

use crate::sat_types::{Lit, Var};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::portfolio::config::EngineConfig;
    use crate::portfolio::factory::ic3_small_circuit;

    #[test]
    fn test_ic3_config_small_circuit_mode_default_is_false() {
        // #4259: ensure the default leaves small_circuit_mode off so existing
        // portfolio configs keep their current domain-restricted SAT behavior.
        let cfg = Ic3Config::default();
        assert!(
            !cfg.small_circuit_mode,
            "small_circuit_mode default must be false to preserve existing behavior"
        );
    }

    #[test]
    fn test_ic3_small_circuit_factory_enables_small_circuit_mode() {
        // #4259: ic3_small_circuit() portfolio config must set
        // small_circuit_mode=true so ay-sat falls back to plain BCP on the
        // Tier 1 HWMCC benchmarks (cal14/cal42/loopv3/microban_1_UNSAT).
        let engine = ic3_small_circuit();
        match engine {
            EngineConfig::Ic3Configured { config, name } => {
                assert_eq!(name, "ic3-small-circuit");
                assert!(
                    config.small_circuit_mode,
                    "ic3_small_circuit() must enable small_circuit_mode for #4259"
                );
            }
            _ => panic!("ic3_small_circuit() must return Ic3Configured variant"),
        }
    }

    #[test]
    fn test_ic3_config_small_circuit_mode_is_independent_flag() {
        // small_circuit_mode is a pure behavioral flag — flipping it should
        // not touch any of the other config fields. Regression guard for
        // spooky coupling in Default / struct update.
        let base = Ic3Config::default();
        let toggled = Ic3Config {
            small_circuit_mode: true,
            ..base.clone()
        };
        assert!(toggled.small_circuit_mode);
        assert_eq!(toggled.ctp, base.ctp);
        assert_eq!(toggled.inf_frame, base.inf_frame);
        assert_eq!(toggled.internal_signals, base.internal_signals);
        assert_eq!(toggled.ctg_max, base.ctg_max);
        assert_eq!(toggled.ctg_limit, base.ctg_limit);
        assert_eq!(toggled.circuit_adapt, base.circuit_adapt);
        assert_eq!(toggled.random_seed, base.random_seed);
        assert_eq!(toggled.parent_lemma, base.parent_lemma);
        assert_eq!(toggled.parent_lemma_mic, base.parent_lemma_mic);
    }
}
