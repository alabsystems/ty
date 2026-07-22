// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Fused 4-lane thread orchestrator for cooperative BFS+BMC+PDR+k-Induction verification.
//!
//! Unlike the [`portfolio`](super::portfolio) orchestrator which races
//! independent lanes, the fused orchestrator enables **cooperative**
//! cross-engine communication via [`SharedCooperativeState`]:
//!
//! - BFS sends concrete frontier states to seed BMC symbolic exploration
//! - PDR-proved invariants let BFS skip per-state invariant checks
//! - k-Induction proves safety via bounded inductive arguments
//! - Any lane's definitive verdict terminates all others via [`SharedVerdict`]
//!
//! # Architecture
//!
//! ```text
//! ┌─────────┐  frontier samples  ┌─────────┐
//! │  Lane 1 │ ─────────────────▶ │  Lane 2 │
//! │   BFS   │                    │   BMC   │
//! └────┬────┘                    └─────────┘
//!      │         ┌─────────┐
//!      │◀───────│  Lane 3 │  invariants_proved
//!               │   PDR   │
//!               └─────────┘
//!                ┌─────────┐
//!                │  Lane 4 │  k-inductive proofs
//!                │  k-Ind  │
//!                └─────────┘
//! ```
//!
//! Part of #3769, #3844.

#[cfg(feature = "ay")]
use std::sync::Arc;

use tla_core::ast::Module;
#[cfg(feature = "ay")]
use tla_core::ast::{OperatorDef, Unit};

#[cfg(feature = "ay")]
use crate::check::wavefront::{entropy_score, WavefrontCompressor, MIN_ENTROPY_THRESHOLD};
use crate::check::{CheckResult, ModelChecker};
use crate::config::Config;
#[cfg(feature = "ay")]
use crate::cooperative_state::SharedCooperativeState;
#[cfg(feature = "ay")]
use crate::eval::EvalCtx;
#[cfg(feature = "ay")]
use crate::shared_verdict::{SharedVerdict, Verdict};

/// Find an operator definition by name in a parsed module's units.
///
/// Scans the module's top-level units for an `Operator` unit whose name
/// matches the given name. Used by the fused orchestrator to extract
/// the Next relation before the full model checker is initialized.
///
/// Part of #3784.
#[cfg(feature = "ay")]
fn find_operator_def(module: &Module, name: &str) -> Option<OperatorDef> {
    module.units.iter().find_map(|unit| match &unit.node {
        Unit::Operator(def) if def.name.node == name => Some(def.clone()),
        _ => None,
    })
}

/// Convenience function: run fused cooperative verification on a spec.
///
/// This is the public API for CLI and library consumers. It creates a
/// `FusedOrchestrator` and runs the 4-lane cooperative verification.
///
/// Part of #3770.
pub fn run_fused_check<'a>(
    module: &'a Module,
    checker_modules: &[&'a Module],
    config: &'a Config,
) -> FusedResult {
    FusedOrchestrator::new(module, checker_modules, config).run()
}

/// Per-checker setup that the CLI applies to the explicit-state checker and that
/// the fused BFS lane must mirror for output parity. Without this, the fused BFS
/// `ModelChecker` is built inside the orchestrator with none of the CLI's
/// configuration: action-label byte offsets never resolve to `line N, col N`
/// (no file-id→path registration) and `--mmap-fingerprints` storage stats are
/// absent (no fingerprint backend). Default is empty (no extra configuration).
#[derive(Default)]
pub struct FusedCheckerConfig {
    /// FileId → source path, registered so trace action labels render as
    /// `line N, col N to line N, col N of module` instead of `bytes A-B`.
    pub file_paths: Vec<(tla_core::FileId, std::path::PathBuf)>,
    /// Optional fingerprint storage backend (e.g. mmap), so the fused BFS lane
    /// uses the same storage as the explicit path and reports its stats.
    pub fingerprint_storage: Option<std::sync::Arc<dyn crate::FingerprintSet>>,
    /// `--max-states` (0 = unbounded). Without this the fused BFS lane explored
    /// unboundedly, ignoring the user's explicit state bound.
    pub max_states: usize,
    /// `--max-depth` (0 = unbounded).
    pub max_depth: usize,
    /// Memory limit in bytes (0 = leave the checker default).
    pub memory_limit_bytes: usize,
    /// Disk limit in bytes (0 = leave the checker default).
    pub disk_limit_bytes: usize,
    /// `--continue-on-error`.
    pub continue_on_error: bool,
    /// Whether to store full states (vs fp-only).
    pub store_states: bool,
}

/// Like [`run_fused_check`] but applies the CLI's per-checker [`FusedCheckerConfig`]
/// to the fused BFS lane so its output matches the explicit-state path.
pub fn run_fused_check_with_config<'a>(
    module: &'a Module,
    checker_modules: &[&'a Module],
    config: &'a Config,
    checker_config: FusedCheckerConfig,
) -> FusedResult {
    FusedOrchestrator::new(module, checker_modules, config)
        .with_checker_config(checker_config)
        .run()
}

/// Which lane produced the first definitive result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedWinner {
    /// BFS explicit-state model checking resolved first.
    Bfs,
    /// BMC symbolic bug finding resolved first.
    Bmc,
    /// PDR (IC3) symbolic safety proving resolved first.
    Pdr,
    /// k-Induction symbolic safety proving resolved first. Part of #3844.
    KInduction,
}

/// Tracks graceful degradation when ay translation fails for one or more
/// symbolic lanes. Part of #3837, #3844.
#[derive(Debug, Clone, Default)]
pub struct SymbolicDegradation {
    /// Whether the BMC lane degraded (translation error or panic).
    pub bmc_degraded: bool,
    /// Human-readable reason for BMC degradation, if any.
    pub bmc_reason: Option<String>,
    /// The raw error message from BMC, if it returned Err. Part of #3837.
    pub bmc_error: Option<String>,
    /// Whether the PDR lane degraded (translation error or panic).
    pub pdr_degraded: bool,
    /// Human-readable reason for PDR degradation, if any.
    pub pdr_reason: Option<String>,
    /// The raw error message from PDR, if it returned Err. Part of #3837.
    pub pdr_error: Option<String>,
    /// Whether the k-Induction lane degraded (translation error or panic).
    /// Part of #3844.
    pub kinduction_degraded: bool,
    /// Human-readable reason for k-Induction degradation, if any.
    /// Part of #3844.
    pub kinduction_reason: Option<String>,
    /// Unsupported constructs encountered across all symbolic lanes.
    pub unsupported_constructs: Vec<String>,
    /// Total number of spec actions detected. Part of #3837.
    pub actions_total: usize,
    /// Number of actions that are SMT-compatible (translatable). Part of #3837.
    pub actions_smt_compatible: usize,
    /// Names of actions that are NOT SMT-compatible. Part of #3837.
    pub unsupported_action_names: Vec<String>,
}

impl SymbolicDegradation {
    /// True if at least one symbolic lane degraded.
    pub fn any_degraded(&self) -> bool {
        self.bmc_degraded || self.pdr_degraded || self.kinduction_degraded
    }

    /// Fraction of symbolic lanes that operated successfully (0.0 to 1.0).
    ///
    /// With 3 symbolic lanes (BMC, PDR, k-Induction), each contributes 1/3.
    pub fn lane_coverage(&self) -> f64 {
        let total = 3.0_f64;
        let failed = (self.bmc_degraded as u8
            + self.pdr_degraded as u8
            + self.kinduction_degraded as u8) as f64;
        (total - failed) / total
    }

    /// Fraction of spec actions that translated successfully to SMT (0.0 to 1.0).
    ///
    /// Returns 1.0 if no actions were detected (nothing to translate).
    /// Part of #3837.
    pub fn action_coverage(&self) -> f64 {
        if self.actions_total == 0 {
            return 1.0;
        }
        self.actions_smt_compatible as f64 / self.actions_total as f64
    }

    /// Combined symbolic coverage metric (0.0 to 1.0).
    ///
    /// Blends lane-level and action-level coverage. When all lanes succeed
    /// and all actions translate, returns 1.0. When no actions were detected,
    /// falls back to lane coverage only.
    pub fn symbolic_coverage(&self) -> f64 {
        if self.actions_total == 0 {
            return self.lane_coverage();
        }
        (self.lane_coverage() + self.action_coverage()) / 2.0
    }

    /// Build a human-readable summary line for the degradation report.
    pub fn summary(&self) -> Option<String> {
        if !self.any_degraded()
            && self.actions_total > 0
            && self.actions_smt_compatible < self.actions_total
        {
            let constructs_str = if self.unsupported_constructs.is_empty() {
                String::new()
            } else {
                format!(": {}", self.unsupported_constructs.join(", "))
            };
            return Some(format!(
                "Symbolic coverage: {}/{} actions translatable ({} unsupported{constructs_str})",
                self.actions_smt_compatible,
                self.actions_total,
                self.actions_total - self.actions_smt_compatible,
            ));
        }
        if !self.any_degraded() {
            return None;
        }
        let coverage_pct = (self.lane_coverage() * 100.0) as u32;
        let mut parts = Vec::new();
        if self.bmc_degraded {
            if let Some(ref reason) = self.bmc_reason {
                parts.push(format!("BMC: {reason}"));
            } else {
                parts.push("BMC: translation failed".to_string());
            }
        }
        if self.pdr_degraded {
            if let Some(ref reason) = self.pdr_reason {
                parts.push(format!("PDR: {reason}"));
            } else {
                parts.push("PDR: translation failed".to_string());
            }
        }
        if self.kinduction_degraded {
            if let Some(ref reason) = self.kinduction_reason {
                parts.push(format!("k-Induction: {reason}"));
            } else {
                parts.push("k-Induction: translation failed".to_string());
            }
        }
        let constructs_str = if self.unsupported_constructs.is_empty() {
            String::new()
        } else {
            format!(" (unsupported: {})", self.unsupported_constructs.join(", "))
        };
        let action_str = if self.actions_total > 0 {
            format!(
                " ({}/{} actions translatable)",
                self.actions_smt_compatible, self.actions_total,
            )
        } else {
            String::new()
        };
        Some(format!(
            "Symbolic coverage: {coverage_pct}% lanes{action_str} — {}{constructs_str}",
            parts.join("; "),
        ))
    }
}

/// Result of a fused cooperative verification run.
#[derive(Debug)]
pub struct FusedResult {
    /// Which lane resolved the verdict first.
    pub winner: FusedWinner,
    /// The BFS result (always present — BFS always runs).
    pub bfs_result: CheckResult,
    /// The BMC result, if the ay feature is enabled and BMC ran.
    #[cfg(feature = "ay")]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) bmc_result: Option<Result<crate::ay_bmc::BmcResult, crate::ay_bmc::BmcError>>,
    /// The PDR result, if the ay feature is enabled and PDR ran.
    #[cfg(feature = "ay")]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) pdr_result: Option<Result<crate::ay_pdr::PdrResult, crate::ay_pdr::PdrError>>,
    /// AY CHC/PDR proof-replay evidence row from the PDR lane, when it ran.
    #[cfg(feature = "ay")]
    pub pdr_proof_replay_evidence: Option<String>,
    /// Frontend-neutral AY shared-engine metadata evidence consumed by fused lanes.
    #[cfg(feature = "ay")]
    pub ay_shared_engine_evidence: Vec<String>,
    /// The k-Induction result, if the ay feature is enabled and k-Induction ran.
    /// Part of #3844.
    #[cfg(feature = "ay")]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) kinduction_result: Option<
        Result<crate::ay_kinduction::KInductionResult, crate::ay_kinduction::KInductionError>,
    >,
    /// Summary string identifying the first verdict source.
    pub symbolic_summary: Option<String>,
    /// Graceful degradation tracking for symbolic lanes. Part of #3837.
    pub degradation: SymbolicDegradation,
    /// Fraction of spec actions that translated successfully to ay (0.0 to 1.0).
    /// Part of #3837.
    pub symbolic_coverage: f64,
    /// Cross-validation result when a symbolic engine produced a verdict.
    ///
    /// Present when BMC found a counterexample (replayed through BFS evaluator)
    /// or when PDR proved safety (checked against BFS completion status).
    /// `None` when only BFS ran or symbolic engines were inconclusive.
    ///
    /// Part of #3836.
    #[cfg(feature = "ay")]
    pub cross_validation: Option<crate::check::cross_validation::CrossValidationResult>,
}

impl FusedResult {
    /// Borrow the PDR lane's AY CHC proof/replay evidence row, if available.
    #[cfg(feature = "ay")]
    pub fn pdr_proof_replay_evidence(&self) -> Option<&str> {
        self.pdr_proof_replay_evidence.as_deref()
    }

    /// Borrow the shared AY engine metadata rows consumed by fused lanes.
    #[cfg(feature = "ay")]
    pub fn ay_shared_engine_evidence(&self) -> &[String] {
        &self.ay_shared_engine_evidence
    }

    /// Reconcile `bfs_result` against the symbolic lanes' verdict, **fail-closed**.
    ///
    /// Closes the #4 *verdict-masking* gap: the CLI used to derive its
    /// user-facing verdict and exit code *solely* from [`Self::bfs_result`]. A
    /// symbolic lane that resolves the `Violated` race truncates the BFS lane
    /// into a result indistinguishable from a clean Success / LimitReached; the
    /// CLI would then print "No error has been found" and exit 0, silently
    /// dropping a found bug (the worst failure for a verification tool). The
    /// original fix covered only BMC violations; k-Induction base-case
    /// counterexamples and PDR unsafe traces had the identical masking shape
    /// (the k-Induction verdict-masking bug: "No error has been found.
    /// Resolved by: k-Induction" with exit 0 on a really-violated spec).
    ///
    /// This method returns [`ReconciledVerdict::SymbolicViolation`] **only**
    /// when a symbolic bug-finding lane (BMC violation, k-Induction base-case
    /// counterexample, or PDR unsafe trace) won the race (so the shared verdict
    /// is `Violated`) *and* the explicit-state evaluator independently replayed
    /// and confirmed the counterexample (`cross_validation.engine_agrees`).
    /// Promotion is therefore *sound*: the permanent oracle (tla-eval)
    /// re-derived the violation, so this can never turn a genuine Success into
    /// a false positive.
    ///
    /// When a symbolic lane won the `Violated` race but its counterexample is
    /// unconfirmed (no cross-validation row, the oracle disagreed, or the
    /// winning lane's result was lost), the truncated BFS result must NOT be
    /// reported as "no error" either — the race win is what truncated BFS, so a
    /// clean `bfs_result` proves nothing. That shape fails closed to
    /// [`ReconciledVerdict::UnvalidatedSymbolicViolation`]: never promoting an
    /// unvalidated symbolic claim, and never letting a cex-bearing lane yield a
    /// success verdict.
    #[cfg(feature = "ay")]
    pub fn reconcile_masked_violation(&self) -> ReconciledVerdict {
        use crate::check::cross_validation::CrossValidationSource;
        // A violation found by the explicit BFS lane is authoritative and is
        // already reported from `bfs_result` — nothing is masked.
        if matches!(
            self.bfs_result,
            CheckResult::InvariantViolation { .. }
                | CheckResult::PropertyViolation { .. }
                | CheckResult::LivenessViolation { .. }
        ) {
            return ReconciledVerdict::FromBfs;
        }
        // Identify the symbolic lane that won the Violated race with a
        // counterexample. `determine_fused_winner` yields `Bmc` (also the
        // fallback when the cex-bearing result was lost),
        // `KInduction`-with-counterexample, or `Pdr`-with-unsafe exactly and
        // only in the masking shape (shared verdict Violated, `bfs_result`
        // clean). `KInduction`/`Pdr` without a cex-bearing result won the
        // *Satisfied* race — certificate-gated elsewhere, nothing masked here.
        let (lane, source) = match self.winner {
            FusedWinner::Bfs => return ReconciledVerdict::FromBfs,
            FusedWinner::Bmc => ("BMC", CrossValidationSource::Bmc),
            FusedWinner::KInduction => match &self.kinduction_result {
                Some(Ok(crate::ay_kinduction::KInductionResult::Counterexample { .. })) => {
                    ("k-Induction", CrossValidationSource::KInduction)
                }
                _ => return ReconciledVerdict::FromBfs,
            },
            FusedWinner::Pdr => match &self.pdr_result {
                Some(Ok(crate::ay_pdr::PdrResult::Unsafe { .. })) => {
                    ("PDR", CrossValidationSource::Pdr)
                }
                _ => return ReconciledVerdict::FromBfs,
            },
        };
        // Promote ONLY on the explicit evaluator's confirmation of THIS lane's
        // counterexample, so the promotion rests on the sound oracle, not on
        // the SMT search alone. The promoted verdict carries the REPLAY-VALIDATED
        // structured trace and the violated invariant's name, so the CLI can
        // route it through the standard violation pipeline (TLC-parity output,
        // ALIAS transform, --trace-format rendering, JSON counterexample).
        if let Some(cv) = &self.cross_validation {
            if cv.engine_agrees && cv.source_engine == source && cv.trace_length > 0 {
                if let (Some(invariant), Some(trace)) =
                    (cv.violated_invariant.clone(), cv.validated_trace.clone())
                {
                    return ReconciledVerdict::SymbolicViolation {
                        lane,
                        detail: cv.detail.clone(),
                        invariant: Some(invariant),
                        trace,
                    };
                }
                // Confirmation row without the replayed payload (should not
                // happen for lane-produced rows) — fall through to fail closed.
            }
        }
        // FAIL CLOSED: a symbolic lane won the Violated race — truncating the
        // BFS lane, whose clean result is therefore NOT authoritative — but the
        // counterexample was not confirmed. Refuse to report "no error".
        ReconciledVerdict::UnvalidatedSymbolicViolation {
            lane,
            detail: match &self.cross_validation {
                Some(cv) => cv.detail.clone(),
                None => format!(
                    "the {lane} lane resolved the verdict race with a counterexample that \
                     was never cross-validated (lane result unavailable)"
                ),
            },
        }
    }

    /// Non-`ay` builds run BFS only — there is no symbolic lane that could mask
    /// a violation, so BFS is always authoritative.
    #[cfg(not(feature = "ay"))]
    pub fn reconcile_masked_violation(&self) -> ReconciledVerdict {
        ReconciledVerdict::FromBfs
    }
}

/// Reconciled, fail-closed verdict view of a fused run, accounting for symbolic
/// lanes that found a violation the BFS lane did not. See
/// [`FusedResult::reconcile_masked_violation`].
#[derive(Debug, Clone)]
pub enum ReconciledVerdict {
    /// `bfs_result` is authoritative — report it as-is (the common case).
    FromBfs,
    /// A racing lane found a violation that `bfs_result` does not reflect, and
    /// the counterexample is trustworthy (explicit-evaluator-confirmed for
    /// symbolic lanes; interpreter-executed for the random walk). The CLI MUST
    /// report a violation (exit 1), never "No error".
    SymbolicViolation {
        /// Which lane found the confirmed counterexample
        /// (e.g. "BMC", "k-Induction", "PDR", "Random walk").
        lane: &'static str,
        /// Human-readable cross-validation detail for the report.
        detail: String,
        /// Name of the violated invariant, when the violation is an invariant
        /// violation (`None` e.g. for a random-walk deadlock). With a
        /// non-empty `trace` this lets the CLI substitute a standard
        /// `CheckResult::InvariantViolation` and reuse the entire normal
        /// violation-reporting pipeline.
        invariant: Option<String>,
        /// Structured, replay-validated counterexample trace.
        /// May be empty when the winning lane's trace was lost.
        trace: crate::check::Trace,
    },
    /// A symbolic lane won the `Violated` race — truncating the BFS lane, so a
    /// clean `bfs_result` proves nothing — but its counterexample could NOT be
    /// re-validated by the explicit-state evaluator (spurious model, evaluator
    /// gap, or the lane result was lost). FAIL CLOSED: the CLI MUST report an
    /// inconclusive verdict (non-zero exit), never "No error" and never an
    /// unvalidated violation.
    UnvalidatedSymbolicViolation {
        /// Which symbolic lane won the race with the unconfirmed counterexample.
        lane: &'static str,
        /// Human-readable cross-validation detail for the report.
        detail: String,
    },
}

// Manual PartialEq: `Trace` carries `ActionLabel`s (no `PartialEq`); compare
// the semantic payload — variant, lane, detail, invariant, and trace states.
impl PartialEq for ReconciledVerdict {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ReconciledVerdict::FromBfs, ReconciledVerdict::FromBfs) => true,
            (
                ReconciledVerdict::SymbolicViolation {
                    lane: l1,
                    detail: d1,
                    invariant: i1,
                    trace: t1,
                },
                ReconciledVerdict::SymbolicViolation {
                    lane: l2,
                    detail: d2,
                    invariant: i2,
                    trace: t2,
                },
            ) => l1 == l2 && d1 == d2 && i1 == i2 && t1.states == t2.states,
            (
                ReconciledVerdict::UnvalidatedSymbolicViolation {
                    lane: l1,
                    detail: d1,
                },
                ReconciledVerdict::UnvalidatedSymbolicViolation {
                    lane: l2,
                    detail: d2,
                },
            ) => l1 == l2 && d1 == d2,
            _ => false,
        }
    }
}

impl Eq for ReconciledVerdict {}

/// Fused 4-lane orchestrator for cooperative multi-engine model checking.
///
/// Spawns BFS, BMC, PDR, and k-Induction lanes in parallel with cross-engine
/// communication via [`SharedCooperativeState`]. Unlike portfolio mode,
/// the lanes actively cooperate: BFS feeds concrete states to BMC,
/// PDR proofs prune BFS invariant checks, and k-Induction provides
/// bounded inductive safety proofs.
pub(crate) struct FusedOrchestrator<'a> {
    module: &'a Module,
    checker_modules: Vec<&'a Module>,
    config: &'a Config,
    checker_config: FusedCheckerConfig,
}

impl<'a> FusedOrchestrator<'a> {
    /// Create a new fused orchestrator for the given spec.
    pub(crate) fn new(
        module: &'a Module,
        checker_modules: &[&'a Module],
        config: &'a Config,
    ) -> Self {
        Self {
            module,
            checker_modules: checker_modules.to_vec(),
            config,
            checker_config: FusedCheckerConfig::default(),
        }
    }

    /// Attach the CLI's per-checker configuration (file registration, storage
    /// backend) so the fused BFS lane is set up like the explicit checker.
    pub(crate) fn with_checker_config(mut self, checker_config: FusedCheckerConfig) -> Self {
        self.checker_config = checker_config;
        self
    }

    /// Run the 4-lane cooperative verification.
    ///
    /// Spawns four threads via [`std::thread::scope`]:
    /// 1. **BFS** — explicit-state model checking, publishing frontier samples
    /// 2. **BMC** — ay bounded model checking, seeded by BFS frontier (ay feature)
    /// 3. **PDR** — ay IC3 safety proving, feeding invariant proofs back (ay feature)
    /// 4. **k-Induction** — ay bounded inductive safety proving (ay feature)
    ///
    /// The first lane to reach a definitive verdict (Satisfied or Violated)
    /// publishes to [`SharedVerdict`]; other lanes exit on their next poll.
    #[cfg(feature = "ay")]
    pub(crate) fn run(&self) -> FusedResult {
        // Part of #3784: detect actions from the Next relation to size
        // per-action metrics correctly before threads start.
        let detected_actions = self
            .config
            .next
            .as_ref()
            .and_then(|next_name| find_operator_def(self.module, next_name))
            .map(|next_def| crate::coverage::detect_actions(&next_def))
            .unwrap_or_default();
        let action_count = detected_actions.len();
        // Part of #3773: pass invariant count for per-invariant proof tracking.
        let invariant_count = self.config.invariants.len();
        let mut coop_state =
            SharedCooperativeState::with_invariant_count(action_count, invariant_count);

        // Part of #3954: expand action expressions before SMT compat check.
        //
        // The raw action expressions from detect_actions() are often just Ident
        // nodes (operator references), which trivially pass the SMT compat filter.
        // To accurately predict whether ay can translate an action, we expand
        // operator definitions first via expand_operators_for_chc, then run the
        // SMT compat check on the fully-expanded expression tree.
        //
        // This changes the oracle from "always accepts" to "accurately rejects
        // actions with Lambda/temporal/InstanceExpr in their expanded body,"
        // improving routing accuracy and reducing ay translator failures.
        let mut compat_ctx = EvalCtx::new();
        compat_ctx.load_module(self.module);
        for extra in &self.checker_modules {
            compat_ctx.load_module(extra);
        }
        let smt_flags: Vec<bool> = detected_actions
            .iter()
            .map(|action| {
                let expanded = crate::ay_pdr::expand_operators_for_chc(
                    &compat_ctx,
                    &action.expr,
                    true, // allow_primed: action exprs contain x' = ... patterns
                );
                crate::cooperative_state::is_expr_smt_compatible(&expanded)
            })
            .collect();
        for (i, &compatible) in smt_flags.iter().enumerate() {
            coop_state.mark_smt_compatible(i, compatible);
        }

        // Part of #3954: log per-action SMT compatibility after expansion.
        let smt_count = smt_flags.iter().filter(|&&f| f).count();
        if action_count > 0 {
            telemetry_eprintln!(
                "[fused] SMT compatibility (expanded): {smt_count}/{action_count} actions compatible"
            );
            for (action, &compatible) in detected_actions.iter().zip(smt_flags.iter()) {
                if !compatible {
                    telemetry_eprintln!("[fused]   incompatible: {}", action.name);
                }
            }
        }

        // Part of #3826: detect exponential state space patterns (e.g., nested
        // SUBSET(SUBSET ...)) and log them. The evaluator's nested powerset
        // optimization (set_construction.rs) rewrites patterns like
        //   {E \in SUBSET(SUBSET Nodes) : \A e \in E : Cardinality(e) = 2}
        // to SUBSET({e \in SUBSET Nodes : Cardinality(e) = 2}), reducing
        // the doubly-exponential 2^(2^N) to a manageable 2^C(N,2).
        //
        // We no longer force all actions to SymbolicOnly here because that
        // makes BFS defer entirely (skipping its init enumeration). With the
        // evaluator optimization, BFS can handle these patterns. Instead, we
        // use normal oracle routing so BFS runs with its optimization while
        // symbolic engines (BMC/PDR/k-Induction) also run in parallel.
        let exponential_complexity =
            crate::check::oracle::detect_exponential_complexity(self.module);
        if let Some(ref signal) = exponential_complexity {
            telemetry_eprintln!("[fused] exponential complexity detected: {}", signal.reason);
            telemetry_eprintln!(
                "[fused] evaluator optimization should reduce state space; \
                 BFS + symbolic engines running in parallel"
            );
        }
        // Part of #3785: initialize oracle with SMT compatibility flags so
        // the initial routing decisions are correct from the first BFS level.
        // Without this, all actions default to `BfsOnly` until the first
        // reroute at level 5, missing the opportunity to route SMT-compatible
        // actions to `Either` from the start.
        coop_state.initialize_oracle(&smt_flags);

        // Register action names for name-based oracle lookups.
        let action_names: Vec<String> = detected_actions.iter().map(|a| a.name.clone()).collect();
        coop_state.register_action_names(&action_names);

        let coop = Arc::new(coop_state);

        // ------------------------------------------------------------------
        // VERDICT-LATENCY GATE (the DieHard 216s defect): the symbolic lanes
        // (BMC / PDR / k-Induction) run as DETACHED workers that report over
        // channels, NOT as scoped threads the orchestrator must join. A
        // scoped join meant one lane stuck *inside* a non-converging solver
        // call held the whole fused run hostage long after the explicit BFS
        // lane had already produced the definitive, user-facing verdict
        // (DieHard: BFS violation in 0.03s, run returned after 211s).
        //
        // The collection loop below waits for lane results only until a
        // definitive verdict exists AND the lane whose result that verdict
        // needs has delivered; the rest get a bounded grace window (they are
        // also interrupted cooperatively via their registered solver
        // interrupt handles + the PDR cancellation token) and are then
        // ABANDONED: their eventual results are discarded, never consumed.
        // Detached workers own clones of the module/config (no scope borrows)
        // so leaving them running is sound; they exit at their next poll /
        // solver control check, and at the latest when the process exits.
        // ------------------------------------------------------------------
        let module_owned = Arc::new(self.module.clone());
        let checker_modules_owned: Arc<Vec<Module>> =
            Arc::new(self.checker_modules.iter().map(|m| (*m).clone()).collect());
        let config_owned = Arc::new(self.config.clone());

        // Cancellation token for the PDR lane's CHC engines: every ay-chc
        // engine main loop polls it, so cancelling stops an in-flight PDR
        // solve (including the detached solve worker inside
        // `solve_pdr_interruptible`) instead of letting it burn CPU to its
        // 300s solve_timeout after the race is already over.
        let pdr_cancel = tla_ay::CancellationToken::new();

        type BmcLaneResult = Result<crate::ay_bmc::BmcResult, crate::ay_bmc::BmcError>;
        type PdrLaneResult = Result<crate::ay_pdr::PdrRunResult, crate::ay_pdr::PdrError>;
        type KindLaneResult =
            Result<crate::ay_kinduction::KInductionResult, crate::ay_kinduction::KInductionError>;

        // `None` payload = the lane panicked (worker cancelled the verdict,
        // preserving the #3867 no-deadlock contract, and reported here).
        let (bmc_tx, bmc_rx) = std::sync::mpsc::channel::<Option<BmcLaneResult>>();
        let (pdr_tx, pdr_rx) = std::sync::mpsc::channel::<Option<PdrLaneResult>>();
        let (kind_tx, kind_rx) = std::sync::mpsc::channel::<Option<KindLaneResult>>();

        let is_exponential = exponential_complexity.is_some();

        // Lane 2: BMC symbolic bug finding (detached).
        //
        // Part of #3772: in normal mode, uses cooperative BMC
        // (check_bmc_cooperative) which polls the frontier channel for
        // concrete states sent by BFS at level boundaries.
        //
        // Part of #3826: when exponential complexity is detected, BFS defers
        // and sends no frontier states — use standalone BMC from Init.
        {
            let coop_bmc = coop.clone();
            let module_arc = module_owned.clone();
            let mods_arc = checker_modules_owned.clone();
            let config_arc = config_owned.clone();
            let handle = std::thread::Builder::new()
                .name("ty-fused-bmc".to_string())
                .spawn(move || {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let module: &Module = &module_arc;
                        let checker_modules: Vec<&Module> = mods_arc.iter().collect();
                        let config: &Config = &config_arc;
                        let mut bmc_ctx = EvalCtx::new();
                        bmc_ctx.load_module(module);
                        // Load EXTENDS-inherited operators and register their
                        // VARIABLES so MC-wrapper specs are symbolically
                        // checkable instead of bailing with "No state
                        // variables declared".
                        for m in &checker_modules {
                            bmc_ctx.load_module(m);
                        }
                        crate::ay_shared::register_state_vars_from_modules(
                            &mut bmc_ctx,
                            module,
                            &checker_modules,
                        );
                        let bmc_config = crate::ay_bmc::BmcConfig {
                            max_depth: 20,
                            ..Default::default()
                        };
                        if is_exponential {
                            // Standalone BMC from Init — no BFS frontier dependency.
                            telemetry_eprintln!(
                                "[fused] BMC running standalone from Init (exponential complexity mode)"
                            );
                            crate::ay_bmc::check_bmc_with_portfolio(
                                module,
                                config,
                                &bmc_ctx,
                                bmc_config,
                                Some(coop_bmc.verdict_arc()),
                            )
                        } else {
                            // Cooperative BMC seeded by BFS frontier states.
                            crate::ay_bmc::check_bmc_cooperative(
                                module,
                                config,
                                &bmc_ctx,
                                bmc_config,
                                coop_bmc.clone(),
                            )
                        }
                    }));
                    // The BMC lane is the sole consumer of frontier samples and
                    // wavefronts: signal exit so the compressor thread and the
                    // BFS frontier sampler stop producing seeds for nobody —
                    // the fused-mode CPU waste when translation degrades
                    // instantly on unsupported operators.
                    coop_bmc.mark_bmc_lane_done();
                    match result {
                        Ok(res) => {
                            let _ = bmc_tx.send(Some(res));
                        }
                        Err(_) => {
                            // Part of #3867: cancel the verdict on panic so
                            // surviving lanes exit instead of spinning forever.
                            coop_bmc.verdict.cancel();
                            let _ = bmc_tx.send(None);
                        }
                    }
                })
                .expect("failed to spawn fused BMC lane thread");
            // Track full thread termination (Rp non-atomic soundness witness).
            coop.register_aux_lane_handle(handle);
        }

        // Lane 3: PDR symbolic safety proving (detached). Part of #3772.
        {
            let coop_pdr = coop.clone();
            let module_arc = module_owned.clone();
            let mods_arc = checker_modules_owned.clone();
            let config_arc = config_owned.clone();
            let pdr_cancel_lane = pdr_cancel.clone();
            let handle = std::thread::Builder::new()
                .name("ty-fused-pdr".to_string())
                .spawn(move || {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let module: &Module = &module_arc;
                        let checker_modules: Vec<&Module> = mods_arc.iter().collect();
                        let config: &Config = &config_arc;
                        let mut pdr_ctx = EvalCtx::new();
                        pdr_ctx.load_module(module);
                        for m in &checker_modules {
                            pdr_ctx.load_module(m);
                        }
                        crate::ay_shared::register_state_vars_from_modules(
                            &mut pdr_ctx,
                            module,
                            &checker_modules,
                        );
                        let mut pdr_config: tla_ay::PdrConfig = Default::default();
                        pdr_config.solve_timeout = Some(std::time::Duration::from_secs(300));
                        // Cooperative teardown: cancelled by the orchestrator
                        // the moment a definitive verdict exists.
                        pdr_config.cancellation_token = Some(pdr_cancel_lane);
                        crate::ay_pdr::check_pdr_cooperative_with_evidence(
                            module,
                            config,
                            &pdr_ctx,
                            pdr_config,
                            coop_pdr.clone(),
                        )
                    }));
                    match result {
                        Ok(res) => {
                            let _ = pdr_tx.send(Some(res));
                        }
                        Err(_) => {
                            coop_pdr.verdict.cancel();
                            let _ = pdr_tx.send(None);
                        }
                    }
                })
                .expect("failed to spawn fused PDR lane thread");
            // Track full thread termination (Rp non-atomic soundness witness).
            coop.register_aux_lane_handle(handle);
        }

        // Lane 4: k-Induction symbolic safety proving (detached). Part of #3844.
        //
        // On success (UNSAT at inductive step), publishes Satisfied to the
        // cooperative verdict. Base-case counterexamples publish Violated
        // (real violations from Init). Inconclusive results publish nothing.
        {
            let coop_kind = coop.clone();
            let module_arc = module_owned.clone();
            let mods_arc = checker_modules_owned.clone();
            let config_arc = config_owned.clone();
            let handle = std::thread::Builder::new()
                .name("ty-fused-kind".to_string())
                .spawn(move || {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let module: &Module = &module_arc;
                        let checker_modules: Vec<&Module> = mods_arc.iter().collect();
                        let config: &Config = &config_arc;
                        let mut kind_ctx = EvalCtx::new();
                        kind_ctx.load_module(module);
                        for m in &checker_modules {
                            kind_ctx.load_module(m);
                        }
                        crate::ay_shared::register_state_vars_from_modules(
                            &mut kind_ctx,
                            module,
                            &checker_modules,
                        );
                        let kind_config = crate::ay_kinduction::KInductionConfig::default();
                        crate::ay_kinduction::check_kinduction_cooperative(
                            module,
                            config,
                            &kind_ctx,
                            kind_config,
                            coop_kind.clone(),
                        )
                    }));
                    match result {
                        Ok(res) => {
                            let _ = kind_tx.send(Some(res));
                        }
                        Err(_) => {
                            coop_kind.verdict.cancel();
                            let _ = kind_tx.send(None);
                        }
                    }
                })
                .expect("failed to spawn fused k-Induction lane thread");
            // Track full thread termination (Rp non-atomic soundness witness).
            coop.register_aux_lane_handle(handle);
        }

        // Wavefront compressor thread (Part of #3794 Wave 3, #3845 Wave 5).
        //
        // Drains frontier samples from the BFS lane, groups by depth,
        // applies entropy-based quality filtering before compressing into
        // a disjunctive formula for the BMC lane.
        //
        // Quality control (#3845): low-entropy batches (identical or
        // near-identical states from early BFS levels) are skipped
        // because they add no diversity for symbolic exploration.
        //
        // Spawned DETACHED (like the symbolic lanes) rather than scoped so its
        // JoinHandle can be registered on the cooperative state before the BFS
        // lane starts: the BFS lane's `TY_RP_VALUE=1` non-atomic-refcount fast
        // path engages only after EVERY registered auxiliary thread has fully
        // terminated, and this thread must be part of that set. It owns only an
        // `Arc` of the cooperative state (no scope borrows), exits when the
        // verdict resolves / BFS completes / the BMC consumer is gone (all
        // signalled on every BFS exit path), and is joined either by
        // `aux_lanes_terminated` or, at the latest, abandoned at process exit —
        // exactly the detached-lane contract documented above.
        {
            let coop_wavefront = coop.clone();
            let handle = std::thread::Builder::new()
                .name("ty-fused-wavefront".to_string())
                .spawn(move || {
                    use std::time::Duration;

                    let compressor = WavefrontCompressor::with_default_threshold();
                    let poll = Duration::from_millis(250);

                    let mut batch: Vec<crate::cooperative_state::FrontierSample> = Vec::new();
                    let mut current_depth: Option<usize> = None;

                    // Helper closure: evaluate entropy, filter, and compress a batch.
                    // Metrics are recorded on the cooperative state so they are observable
                    // externally (not lost when this thread exits). Part of #3794.
                    let try_compress_batch =
                        |batch: &[crate::cooperative_state::FrontierSample],
                         compressor: &WavefrontCompressor,
                         coop: &SharedCooperativeState|
                         -> bool {
                            if !compressor.should_compress(batch.len()) {
                                return false;
                            }

                            let score = entropy_score(batch);

                            if score < MIN_ENTROPY_THRESHOLD {
                                coop.record_wavefront_skipped_low_entropy();
                                return false;
                            }

                            if let Some(formula) = compressor.compress_frontier(batch) {
                                if coop.send_wavefront(formula) {
                                    coop.record_wavefront_sent();
                                    return true;
                                }
                            }
                            false
                        };

                    loop {
                        // Exit when the racing verdict is resolved OR when BFS has
                        // finished for ANY reason (Success, Violation, Deadlock,
                        // depth-limit). A reached deadlock publishes no verdict
                        // (CheckResult::Deadlock maps to Verdict::Unknown, a no-op),
                        // so without the is_bfs_complete() check this compressor thread
                        // would spin forever — the fused-mode deadlock hang. Mirrors
                        // the BMC lane fix (#4002).
                        //
                        // Also exit when the BMC lane — the sole wavefront consumer —
                        // has already exited (e.g., instant translation degradation
                        // on unsupported operators): compressing frontiers nobody
                        // will read is pure CPU waste for the rest of the BFS run.
                        if coop_wavefront.is_resolved()
                            || coop_wavefront.is_bfs_complete()
                            || coop_wavefront.is_bmc_lane_done()
                        {
                            try_compress_batch(&batch, &compressor, &coop_wavefront);
                            return;
                        }

                        match coop_wavefront.recv_frontier_sample(poll) {
                            Some(sample) => {
                                let sample_depth = sample.depth;

                                // Depth transition: compress accumulated batch.
                                if let Some(prev) = current_depth {
                                    if sample_depth > prev && !batch.is_empty() {
                                        try_compress_batch(&batch, &compressor, &coop_wavefront);
                                        batch.clear();
                                    }
                                }

                                current_depth = Some(sample_depth);
                                batch.push(sample);
                            }
                            None => {
                                // Timeout: if we have a large enough batch, compress
                                // it even without a depth transition (BFS may have
                                // stalled or finished the level).
                                if compressor.should_compress(batch.len()) {
                                    try_compress_batch(&batch, &compressor, &coop_wavefront);
                                    batch.clear();
                                }
                            }
                        }
                    }
                })
                .expect("failed to spawn fused wavefront compressor thread");
            coop.register_aux_lane_handle(handle);
        }

        // ALL auxiliary threads (BMC/PDR/k-Induction lanes + wavefront
        // compressor) are now registered; seal BEFORE the BFS lane spawns so
        // its termination poll can never observe an incomplete handle set.
        coop.seal_aux_lane_registration();

        std::thread::scope(|scope| {
            let coop_bfs = coop.clone();

            let module = self.module;
            let checker_modules = &self.checker_modules;
            let config = self.config;
            let checker_config = &self.checker_config;

            // Lane 1: BFS explicit-state model checking.
            //
            // Part of #3772: wire cooperative state into BFS so that:
            // (a) frontier sampler sends concrete states to BMC via the channel,
            // (b) BFS polls the cooperative verdict for early termination when
            //     BMC/PDR resolve first.
            // Part of #3823: pass checker_modules for EXTENDS/INSTANCE support.
            // Part of #3867: catch_unwind on all lanes to cancel verdict on panic,
            // preventing deadlock where surviving lanes spin forever.
            let coop_bfs_panic = coop.clone();
            let bfs_handle = scope.spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut checker =
                        ModelChecker::new_with_extends(module, checker_modules, config);
                    // Mirror the CLI's per-checker configuration so fused output
                    // matches the explicit path: register file paths (so action
                    // labels resolve to line:col) and attach the fingerprint
                    // storage backend (so --mmap-fingerprints stats appear).
                    for (file_id, path) in &checker_config.file_paths {
                        checker.register_file_path(*file_id, path.clone());
                    }
                    if let Some(ref storage) = checker_config.fingerprint_storage {
                        checker.set_fingerprint_storage(storage.clone());
                    }
                    // Honor the explicit exploration budget (--max-states /
                    // --max-depth / memory / disk); the fused lane previously
                    // ignored these and explored unboundedly.
                    if checker_config.max_states > 0 {
                        checker.set_max_states(checker_config.max_states);
                    }
                    if checker_config.max_depth > 0 {
                        checker.set_max_depth(checker_config.max_depth);
                    }
                    if checker_config.memory_limit_bytes > 0 {
                        checker.set_memory_limit(checker_config.memory_limit_bytes);
                    }
                    if checker_config.disk_limit_bytes > 0 {
                        checker.set_disk_limit(checker_config.disk_limit_bytes);
                    }
                    checker.set_continue_on_error(checker_config.continue_on_error);
                    checker.set_store_states(checker_config.store_states);
                    // V2 vacuity gate: enable per-action coverage tracking so the
                    // BFS lane detects dead (never-fired) actions — UNLESS the
                    // native-fused fast path is viable, in which case coverage
                    // collection would make the native level ineligible and force
                    // the ~20x-slower interpreter. `set_default_dead_action_coverage`
                    // skips it (and pairs with eager native compile) for
                    // safety-only trust-cg runs, keeps it on for `--coverage` and
                    // non-native runs, and subsumes the prior strict-mode carve-out.
                    checker.set_default_dead_action_coverage();
                    checker.set_portfolio_verdict(coop_bfs.verdict_arc());
                    checker.set_cooperative_state(coop_bfs.clone());
                    checker.check()
                }));
                if result.is_err() {
                    coop_bfs_panic.verdict.cancel();
                }
                result
            });

            // Join the BFS lane: if BFS panicked, propagate by resuming the
            // panic. BFS is the primary lane — its result is always needed.
            // (The wavefront thread exits on its own: BFS always marks
            // completion, and a BFS panic cancels the verdict.)
            let bfs_result = match bfs_handle.join() {
                Ok(Ok(result)) => result,
                Ok(Err(payload)) => std::panic::resume_unwind(payload),
                Err(join_err) => std::panic::resume_unwind(join_err),
            };

            // ---------------------------------------------------------------
            // Collect the detached symbolic lanes' results — bounded by the
            // verdict-latency gate.
            //
            // Wait states per lane slot: `None` = still pending,
            // `Some(None)` = lane panicked, `Some(Some(r))` = lane delivered.
            //
            // Termination policy (soundness-ordered):
            //  1. All lanes delivered → exactly the pre-existing behavior.
            //  2. A definitive verdict exists AND the lane result that verdict
            //     depends on is secured → interrupt the remaining solvers and
            //     give the stragglers a bounded grace window, then ABANDON
            //     them (verdict-latency fix). "Secured" means:
            //       - Violated: the BFS result is itself a violation (BFS won;
            //         no symbolic confirmation is needed for a sound explicit
            //         counterexample), or a symbolic violating result (BMC
            //         Violation / k-Ind Counterexample / PDR Unsafe) arrived
            //         (needed for winner attribution + cross-validation).
            //       - Satisfied: the BFS result is Success, or a symbolic
            //         proving result (PDR Safe / k-Ind Proved) arrived.
            //       - Cancelled (a lane panicked; `get()` is None): nothing
            //         can be needed from the remaining lanes.
            //     BFS results that are definitive for the user but publish no
            //     verdict (Deadlock / Vacuous / Error map to Verdict::Unknown)
            //     open the same gate: symbolic lanes cannot overturn them.
            //  3. Otherwise (e.g., BFS hit a state limit and no lane resolved
            //     yet): wait for all lanes, exactly as before — a symbolic
            //     lane may still be the one that finds the verdict.
            //
            // An abandoned lane's eventual result is NEVER consumed: the slot
            // is filled with a synthetic inconclusive result (identical in
            // kind to the lane's own cooperative-teardown returns), so no
            // verdict can ever originate from a cancelled/partial lane.
            // ---------------------------------------------------------------
            let mut bmc_slot: Option<Option<BmcLaneResult>> = None;
            let mut pdr_slot: Option<Option<PdrLaneResult>> = None;
            let mut kind_slot: Option<Option<KindLaneResult>> = None;
            {
                use std::sync::mpsc::TryRecvError;
                use std::time::Instant;

                let lane_grace = fused_lane_grace();
                let poll = std::time::Duration::from_millis(10);
                let mut grace_started: Option<Instant> = None;
                let mut solvers_interrupted = false;

                // One macro instead of a closure: the three lanes have
                // distinct payload types.
                macro_rules! drain_slot {
                    ($slot:expr, $rx:expr) => {
                        if $slot.is_none() {
                            match $rx.try_recv() {
                                Ok(v) => $slot = Some(v),
                                Err(TryRecvError::Empty) => {}
                                // Sender dropped without sending: the worker
                                // died outside catch_unwind. Treat as panicked.
                                Err(TryRecvError::Disconnected) => $slot = Some(None),
                            }
                        }
                    };
                }

                loop {
                    drain_slot!(bmc_slot, bmc_rx);
                    drain_slot!(pdr_slot, pdr_rx);
                    drain_slot!(kind_slot, kind_rx);
                    if bmc_slot.is_some() && pdr_slot.is_some() && kind_slot.is_some() {
                        break;
                    }

                    let verdict_resolved = coop.verdict.is_resolved();
                    let bfs_definitive_unpublished = matches!(
                        bfs_result,
                        CheckResult::Deadlock { .. }
                            | CheckResult::Vacuous { .. }
                            | CheckResult::Error { .. }
                    );
                    if verdict_resolved || bfs_definitive_unpublished {
                        if !solvers_interrupted {
                            // Cooperative teardown of in-flight solves: flip
                            // every registered ay solver interrupt handle and
                            // cancel the PDR lane's CHC engines. Best-effort —
                            // the grace window below is the hard bound.
                            coop.interrupt_registered_solvers();
                            pdr_cancel.cancel();
                            solvers_interrupted = true;
                        }
                        let winner_secured = match coop.verdict.get() {
                            Some(Verdict::Violated) => {
                                matches!(
                                    bfs_result,
                                    CheckResult::InvariantViolation { .. }
                                        | CheckResult::PropertyViolation { .. }
                                        | CheckResult::LivenessViolation { .. }
                                ) || matches!(
                                    bmc_slot,
                                    Some(Some(Ok(crate::ay_bmc::BmcResult::Violation { .. })))
                                ) || matches!(
                                    kind_slot,
                                    Some(Some(Ok(
                                        crate::ay_kinduction::KInductionResult::Counterexample {
                                            ..
                                        }
                                    )))
                                ) || matches!(
                                    &pdr_slot,
                                    Some(Some(Ok(run)))
                                        if matches!(run.result, crate::ay_pdr::PdrResult::Unsafe { .. })
                                )
                            }
                            Some(Verdict::Satisfied) => {
                                matches!(bfs_result, CheckResult::Success(_))
                                    || matches!(
                                        &pdr_slot,
                                        Some(Some(Ok(run)))
                                            if matches!(run.result, crate::ay_pdr::PdrResult::Safe { .. })
                                    )
                                    || matches!(
                                        kind_slot,
                                        Some(Some(Ok(
                                            crate::ay_kinduction::KInductionResult::Proved { .. }
                                        )))
                                    )
                            }
                            // Cancelled (panic teardown) or unresolved with a
                            // definitive unpublished BFS result: no symbolic
                            // result is needed for the user-facing verdict.
                            // (`get()` never returns `Some(Unknown)` — the slot
                            // only stores definitive verdicts — but the type
                            // requires the arm.)
                            None | Some(Verdict::Unknown) => true,
                        };
                        if winner_secured {
                            let started = *grace_started.get_or_insert_with(Instant::now);
                            if started.elapsed() >= lane_grace {
                                break;
                            }
                        }
                        // Not secured: the publishing lane's send is imminent
                        // (every lane publishes immediately before returning);
                        // keep draining without starting the grace clock.
                    }
                    std::thread::sleep(poll);
                }
            }

            // Map lane slots to results. Panicked lanes stay `None` (the
            // pre-existing degradation contract); abandoned lanes get a
            // synthetic inconclusive result — same kind the lane itself
            // returns on cooperative teardown — and their real (late) results
            // are discarded, never consumed.
            let bmc_result: Option<BmcLaneResult> = match bmc_slot {
                Some(Some(result)) => Some(result),
                Some(None) => {
                    telemetry_eprintln!("[fused] BMC lane panicked — continuing with BFS result");
                    None
                }
                None => {
                    telemetry_eprintln!(
                        "[fused] BMC lane abandoned after definitive verdict — \
                         its late result will be discarded"
                    );
                    Some(Ok(crate::ay_bmc::BmcResult::Unknown {
                        depth: 0,
                        reason: String::from(
                            "lane abandoned: definitive verdict resolved before BMC completed",
                        ),
                    }))
                }
            };

            let pdr_run_result: Option<PdrLaneResult> = match pdr_slot {
                Some(Some(result)) => Some(result),
                Some(None) => {
                    telemetry_eprintln!("[fused] PDR lane panicked — continuing with BFS result");
                    None
                }
                None => {
                    telemetry_eprintln!(
                        "[fused] PDR lane abandoned after definitive verdict — \
                         its late result will be discarded"
                    );
                    Some(Ok(crate::ay_pdr::unknown_pdr_run_with_missing_evidence(
                        "lane abandoned: definitive verdict resolved before PDR completed",
                    )))
                }
            };
            let pdr_proof_replay_evidence = pdr_run_result
                .as_ref()
                .and_then(|result| result.as_ref().ok())
                .map(|run| run.proof_replay_evidence.clone());
            let pdr_result =
                pdr_run_result.map(|result| result.map(crate::ay_pdr::PdrRunResult::into_result));

            let kinduction_result: Option<KindLaneResult> = match kind_slot {
                Some(Some(result)) => Some(result),
                Some(None) => {
                    telemetry_eprintln!(
                        "[fused] k-Induction lane panicked — continuing with BFS result"
                    );
                    None
                }
                None => {
                    telemetry_eprintln!(
                        "[fused] k-Induction lane abandoned after definitive verdict — \
                         its late result will be discarded"
                    );
                    Some(Ok(crate::ay_kinduction::KInductionResult::Unknown {
                        max_k: 0,
                        reason: String::from(
                            "lane abandoned: definitive verdict resolved before \
                             k-Induction completed",
                        ),
                    }))
                }
            };

            // Part of #3837, #3844: build graceful degradation report from
            // BMC/PDR/k-Induction results and per-action SMT compatibility data.
            let degradation = build_degradation(
                &bmc_result,
                &pdr_result,
                &kinduction_result,
                &smt_flags,
                &action_names,
            );
            if let Some(ref summary) = degradation.summary() {
                telemetry_eprintln!("[fused] {summary}");
            }

            // Part of #3837: per-action symbolic coverage fraction.
            let symbolic_coverage = degradation.action_coverage();

            let winner = determine_fused_winner(
                &coop.verdict,
                &bfs_result,
                &bmc_result,
                &pdr_result,
                &kinduction_result,
            );
            let mut symbolic_summary = match winner {
                FusedWinner::Bfs => "Winner: BFS (explicit-state)".to_string(),
                FusedWinner::Bmc => "Winner: BMC (symbolic bug finding)".to_string(),
                FusedWinner::Pdr => "Winner: PDR (symbolic safety proving)".to_string(),
                FusedWinner::KInduction => {
                    "Winner: k-Induction (symbolic safety proving)".to_string()
                }
            };
            if let Some(deg_summary) = degradation.summary() {
                symbolic_summary.push_str(&format!(". {deg_summary}"));
            }
            let symbolic_summary = Some(symbolic_summary);

            // Part of #3836: cross-validate symbolic verdicts through BFS evaluator.
            let cross_validation = perform_cross_validation(
                module,
                config,
                &winner,
                &bmc_result,
                &pdr_result,
                &kinduction_result,
                &bfs_result,
            );
            if let Some(ref cv) = cross_validation {
                if cv.engine_agrees {
                    telemetry_eprintln!("[fused] cross-validation OK: {}", cv.detail);
                } else {
                    telemetry_eprintln!("[fused] cross-validation WARNING: {}", cv.detail);
                }
            }

            // BMC incremental deepening diagnostics.
            {
                let seeds_completed = coop.bmc_seeds_completed();
                let bmc_depth = coop.bmc_max_depth();
                let avg_depth = coop.bmc_avg_seed_depth();
                let deprioritized = coop.bmc_seeds_deprioritized();
                let wavefronts_consumed = coop.wavefronts_consumed();
                let wavefronts_sent = coop.wavefronts_sent();
                let frontier_hint = coop.bfs_frontier_depth_hint();

                if seeds_completed > 0 || wavefronts_consumed > 0 {
                    telemetry_eprintln!(
                        "[fused] BMC deepening: {} seeds completed, max depth {}, avg depth {:.1}, \
                         {} wavefronts consumed/{} sent, BFS frontier hint depth {}, {} seeds deprioritized",
                        seeds_completed,
                        bmc_depth,
                        avg_depth,
                        wavefronts_consumed,
                        wavefronts_sent,
                        frontier_hint,
                        deprioritized,
                    );
                }
            }

            FusedResult {
                winner,
                bfs_result,
                bmc_result,
                pdr_result,
                pdr_proof_replay_evidence,
                ay_shared_engine_evidence: {
                    let program = crate::check::portfolio::prepared_analytical_portfolio_program(
                        self.module,
                        &self.checker_modules,
                        self.config,
                        tla_mc_core::PreparedProgramPayloadKind::Tla,
                    );
                    crate::ay_shared::ay_shared_engine_metadata_and_admission_evidence_rows(
                        "TLA", &program,
                    )
                },
                kinduction_result,
                symbolic_summary,
                degradation,
                symbolic_coverage,
                cross_validation,
            }
        })
    }

    /// Non-ay fallback: runs BFS only (BMC/PDR/k-Induction require ay feature).
    #[cfg(not(feature = "ay"))]
    pub(crate) fn run(&self) -> FusedResult {
        let mut checker =
            ModelChecker::new_with_extends(self.module, &self.checker_modules, self.config);
        // V2 vacuity gate: track per-action coverage for dead-action detection,
        // matching the explicit-state path.
        checker.set_default_dead_action_coverage();
        let bfs_result = checker.check();

        FusedResult {
            winner: FusedWinner::Bfs,
            bfs_result,
            symbolic_summary: None,
            degradation: SymbolicDegradation::default(),
            symbolic_coverage: 1.0,
        }
    }
}

/// Grace window granted to straggler symbolic lanes once a definitive verdict
/// exists and the result that verdict depends on is already secured.
///
/// Long enough for a healthy lane to reach its next cooperative poll boundary
/// (the cooperative BMC frontier recv blocks up to 500ms; PDR polls every
/// 25ms) so lanes that are merely *between* checks still deliver their real
/// results; short enough that a solver call stuck in a non-converging /
/// non-interruptible phase (the DieHard 216s defect) cannot hold the verdict
/// hostage. Tunable via `TY_FUSED_LANE_GRACE_MS` for diagnosis.
#[cfg(feature = "ay")]
fn fused_lane_grace() -> std::time::Duration {
    static GRACE_MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let ms = *GRACE_MS.get_or_init(|| {
        std::env::var("TY_FUSED_LANE_GRACE_MS")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(1000)
    });
    std::time::Duration::from_millis(ms)
}

/// Classify a BMC error as a translation/degradation failure. Part of #3837.
#[cfg(feature = "ay")]
fn classify_bmc_error(err: &crate::ay_bmc::BmcError) -> (bool, String, Vec<String>) {
    use crate::ay_bmc::BmcError;
    match err {
        BmcError::TranslationError(msg) => {
            let constructs = extract_unsupported_constructs(msg);
            (true, format!("translation failed: {msg}"), constructs)
        }
        BmcError::SolverFailed(msg) => (true, format!("solver failed: {msg}"), vec![]),
        BmcError::MissingSpec(msg) => (false, format!("missing spec: {msg}"), vec![]),
        BmcError::NoInvariants => (false, "no invariants to check".to_string(), vec![]),
        BmcError::CheckError(msg) => (true, format!("check error: {msg}"), vec![]),
    }
}

/// Classify a PDR error as a translation/degradation failure. Part of #3837.
#[cfg(feature = "ay")]
fn classify_pdr_error(err: &crate::ay_pdr::PdrError) -> (bool, String, Vec<String>) {
    use crate::ay_pdr::PdrError;
    match err {
        PdrError::TranslationError(msg) => {
            let constructs = extract_unsupported_constructs(msg);
            (true, format!("translation failed: {msg}"), constructs)
        }
        PdrError::UnsupportedExpr(msg) => {
            let constructs = extract_unsupported_constructs(msg);
            (true, format!("unsupported expression: {msg}"), constructs)
        }
        PdrError::SortInference(msg) => (true, format!("sort inference failed: {msg}"), vec![]),
        PdrError::MissingSpec(msg) => (false, format!("missing spec: {msg}"), vec![]),
        PdrError::NoInvariants => (false, "no invariants to check".to_string(), vec![]),
        PdrError::CheckError(msg) => (true, format!("check error: {msg}"), vec![]),
    }
}

/// Extract names of unsupported TLA+ constructs from error messages. Part of #3837.
///
/// Scans for patterns like "unsupported operator: CHOOSE", "SetFilter not supported", etc.
#[cfg(feature = "ay")]
fn extract_unsupported_constructs(msg: &str) -> Vec<String> {
    let mut constructs = Vec::new();
    let lower = msg.to_lowercase();
    for keyword in &[
        "CHOOSE",
        "SetFilter",
        "SetMap",
        "Lambda",
        "RecursiveOp",
        "SUBSET",
        "UNION",
        "BoundedQuant",
        "LetIn",
    ] {
        if lower.contains(&keyword.to_lowercase()) {
            constructs.push(keyword.to_string());
        }
    }
    if constructs.is_empty() && lower.contains("unsupported") {
        constructs.push("unknown construct".to_string());
    }
    constructs
}

/// Classify a k-Induction error as a translation/degradation failure. Part of #3844.
#[cfg(feature = "ay")]
fn classify_kinduction_error(
    err: &crate::ay_kinduction::KInductionError,
) -> (bool, String, Vec<String>) {
    use crate::ay_kinduction::KInductionError;
    match err {
        KInductionError::TranslationError(msg) => {
            let constructs = extract_unsupported_constructs(msg);
            (true, format!("translation failed: {msg}"), constructs)
        }
        KInductionError::SolverFailed(msg) => (true, format!("solver failed: {msg}"), vec![]),
        KInductionError::MissingSpec(msg) => (false, format!("missing spec: {msg}"), vec![]),
        KInductionError::NoInvariants => (false, "no invariants to check".to_string(), vec![]),
        KInductionError::CheckError(err) => (true, format!("check error: {err:?}"), vec![]),
    }
}

/// Build a `SymbolicDegradation` from the BMC, PDR, and k-Induction results
/// plus per-action SMT compatibility data. Part of #3837, #3844.
#[cfg(feature = "ay")]
fn build_degradation(
    bmc_result: &Option<Result<crate::ay_bmc::BmcResult, crate::ay_bmc::BmcError>>,
    pdr_result: &Option<Result<crate::ay_pdr::PdrResult, crate::ay_pdr::PdrError>>,
    kinduction_result: &Option<
        Result<crate::ay_kinduction::KInductionResult, crate::ay_kinduction::KInductionError>,
    >,
    smt_flags: &[bool],
    action_names: &[String],
) -> SymbolicDegradation {
    let actions_total = smt_flags.len();
    let actions_smt_compatible = smt_flags.iter().filter(|&&f| f).count();
    let unsupported_action_names: Vec<String> = smt_flags
        .iter()
        .zip(action_names.iter())
        .filter(|(&compatible, _)| !compatible)
        .map(|(_, name)| name.clone())
        .collect();

    let mut degradation = SymbolicDegradation {
        actions_total,
        actions_smt_compatible,
        unsupported_action_names,
        ..Default::default()
    };
    match bmc_result {
        None => {
            degradation.bmc_degraded = true;
            degradation.bmc_reason = Some("lane panicked".to_string());
        }
        Some(Err(err)) => {
            let (_d, reason, constructs) = classify_bmc_error(err);
            degradation.bmc_degraded = true;
            degradation.bmc_reason = Some(reason.clone());
            degradation.bmc_error = Some(format!("{err:?}"));
            degradation.unsupported_constructs.extend(constructs);
        }
        Some(Ok(_)) => {}
    }
    match pdr_result {
        None => {
            degradation.pdr_degraded = true;
            degradation.pdr_reason = Some("lane panicked".to_string());
        }
        Some(Err(err)) => {
            let (_d, reason, constructs) = classify_pdr_error(err);
            degradation.pdr_degraded = true;
            degradation.pdr_reason = Some(reason.clone());
            degradation.pdr_error = Some(format!("{err:?}"));
            degradation.unsupported_constructs.extend(constructs);
        }
        Some(Ok(_)) => {}
    }
    // Part of #3844: k-Induction degradation tracking.
    match kinduction_result {
        None => {
            degradation.kinduction_degraded = true;
            degradation.kinduction_reason = Some("lane panicked".to_string());
        }
        Some(Err(err)) => {
            let (_d, reason, constructs) = classify_kinduction_error(err);
            degradation.kinduction_degraded = true;
            degradation.kinduction_reason = Some(reason);
            degradation.unsupported_constructs.extend(constructs);
        }
        Some(Ok(_)) => {}
    }
    degradation.unsupported_constructs.sort();
    degradation.unsupported_constructs.dedup();
    degradation
}

/// Perform cross-validation of a symbolic engine's verdict against the BFS evaluator.
///
/// Called after all lanes have joined and the winner is determined. Returns `Some`
/// when a symbolic engine produced a definitive verdict (BMC violation or PDR safety)
/// and cross-validation is possible. Returns `None` when only BFS ran or symbolic
/// engines were inconclusive.
///
/// Cross-validation is **non-blocking**: if the BFS evaluator panics during replay
/// (e.g., due to an unsupported construct or evaluator bug), `catch_unwind` catches
/// the panic and returns a failed cross-validation result instead of crashing the
/// orchestrator. This ensures the safety net never makes things worse.
///
/// Part of #3836.
#[cfg(feature = "ay")]
fn perform_cross_validation(
    module: &Module,
    config: &Config,
    winner: &FusedWinner,
    bmc_result: &Option<Result<crate::ay_bmc::BmcResult, crate::ay_bmc::BmcError>>,
    pdr_result: &Option<Result<crate::ay_pdr::PdrResult, crate::ay_pdr::PdrError>>,
    kinduction_result: &Option<
        Result<crate::ay_kinduction::KInductionResult, crate::ay_kinduction::KInductionError>,
    >,
    bfs_result: &CheckResult,
) -> Option<crate::check::cross_validation::CrossValidationResult> {
    use crate::check::cross_validation::{
        cross_validate_bmc_trace, cross_validate_pdr_safety, cross_validate_symbolic_trace,
        pdr_trace_to_bmc_states, CrossValidationResult, CrossValidationSource,
    };

    // Helper: wrap a cross-validation call in catch_unwind so a panic in the
    // BFS evaluator during replay does not crash the orchestrator.
    let safe_validate = |f: Box<dyn FnOnce() -> CrossValidationResult>| -> CrossValidationResult {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
            Ok(result) => result,
            Err(payload) => {
                let reason = payload
                    .downcast_ref::<String>()
                    .map(|s| s.as_str())
                    .or_else(|| payload.downcast_ref::<&str>().copied())
                    .unwrap_or("unknown panic");
                telemetry_eprintln!(
                    "[fused] cross-validation panicked — falling back to BFS-only result: {reason}"
                );
                CrossValidationResult {
                    engine_agrees: false,
                    trace_length: 0,
                    source_engine: CrossValidationSource::Bmc,
                    detail: format!(
                        "cross-validation panicked during BFS replay: {reason} — \
                         falling back to BFS-only result"
                    ),
                    violated_invariant: None,
                    validated_trace: None,
                }
            }
        }
    };

    match winner {
        FusedWinner::Bmc => {
            // BMC won -- cross-validate its counterexample trace through BFS evaluator.
            if let Some(Ok(crate::ay_bmc::BmcResult::Violation { trace, .. })) = bmc_result {
                let trace = trace.clone();
                Some(safe_validate(Box::new(move || {
                    cross_validate_bmc_trace(module, config, &trace)
                })))
            } else {
                None
            }
        }
        FusedWinner::Pdr => {
            // PDR won the Satisfied race -- cross-validate its safety proof
            // against BFS completion status. PDR can also win the Violated race
            // (Unsafe): replay the counterexample's final state through the
            // explicit evaluator, exactly like a BMC violation.
            if let Some(Ok(crate::ay_pdr::PdrResult::Safe { invariant })) = pdr_result {
                let invariant = invariant.clone();
                Some(safe_validate(Box::new(move || {
                    cross_validate_pdr_safety(bfs_result, &invariant)
                })))
            } else if let Some(Ok(crate::ay_pdr::PdrResult::Unsafe { trace })) = pdr_result {
                let trace = pdr_trace_to_bmc_states(trace);
                Some(safe_validate(Box::new(move || {
                    cross_validate_symbolic_trace(module, config, &trace, CrossValidationSource::Pdr)
                })))
            } else {
                None
            }
        }
        FusedWinner::KInduction => {
            // k-Induction won -- for a safety proof, cross-validate against BFS
            // completion status; for a base-case counterexample (Violated race
            // win), replay the trace through the explicit evaluator, exactly
            // like a BMC violation. Without the counterexample arm, a k-Ind
            // base-case violation was never cross-validated, so the masked-
            // violation reconciliation could not promote it and the CLI
            // reported the race-truncated BFS "Success" — a real violation
            // masked as "No error" (the k-Induction verdict-masking bug).
            if let Some(Ok(crate::ay_kinduction::KInductionResult::Proved { k })) =
                kinduction_result
            {
                let proved_k = *k;
                Some(safe_validate(Box::new(move || {
                    crate::check::cross_validation::cross_validate_kinduction_safety(
                        bfs_result, proved_k,
                    )
                })))
            } else if let Some(Ok(crate::ay_kinduction::KInductionResult::Counterexample {
                trace,
                ..
            })) = kinduction_result
            {
                let trace = trace.clone();
                Some(safe_validate(Box::new(move || {
                    cross_validate_symbolic_trace(
                        module,
                        config,
                        &trace,
                        CrossValidationSource::KInduction,
                    )
                })))
            } else {
                None
            }
        }
        FusedWinner::Bfs => {
            // BFS won -- optionally cross-validate any symbolic result for observability.
            if let Some(Ok(crate::ay_bmc::BmcResult::Violation { trace, .. })) = bmc_result {
                let trace = trace.clone();
                return Some(safe_validate(Box::new(move || {
                    cross_validate_bmc_trace(module, config, &trace)
                })));
            }
            if let Some(Ok(crate::ay_pdr::PdrResult::Safe { invariant })) = pdr_result {
                let invariant = invariant.clone();
                return Some(safe_validate(Box::new(move || {
                    cross_validate_pdr_safety(bfs_result, &invariant)
                })));
            }
            None
        }
    }
}

/// Determine which lane won by examining the published verdict and
/// correlating with each lane's result. Part of #3844: added k-induction.
///
/// For a `Violated` verdict the attribution scans every cex-bearing lane
/// result (k-Induction base-case counterexample, BMC violation, PDR unsafe) so
/// the masked-violation reconciliation can pair the winner with its
/// counterexample. `Bmc` remains the fallback attribution when the verdict is
/// `Violated` but no lane result carries the counterexample (e.g. the
/// publishing lane panicked after publishing) — the reconciliation then fails
/// closed on the missing cross-validation.
#[cfg(feature = "ay")]
fn determine_fused_winner(
    verdict: &SharedVerdict,
    bfs_result: &CheckResult,
    bmc_result: &Option<Result<crate::ay_bmc::BmcResult, crate::ay_bmc::BmcError>>,
    pdr_result: &Option<Result<crate::ay_pdr::PdrResult, crate::ay_pdr::PdrError>>,
    kinduction_result: &Option<
        Result<crate::ay_kinduction::KInductionResult, crate::ay_kinduction::KInductionError>,
    >,
) -> FusedWinner {
    match verdict.get() {
        Some(Verdict::Satisfied) => match bfs_result {
            CheckResult::Success(_) => FusedWinner::Bfs,
            _ => {
                // If k-induction proved safety, attribute the win to it.
                // Otherwise attribute to PDR (the other safety-proving lane).
                if let Some(Ok(crate::ay_kinduction::KInductionResult::Proved { .. })) =
                    kinduction_result
                {
                    FusedWinner::KInduction
                } else {
                    FusedWinner::Pdr
                }
            }
        },
        Some(Verdict::Violated) => match bfs_result {
            CheckResult::InvariantViolation { .. }
            | CheckResult::PropertyViolation { .. }
            | CheckResult::LivenessViolation { .. } => FusedWinner::Bfs,
            _ => {
                // Attribute to the lane whose result actually carries the
                // counterexample: k-induction base case, then BMC, then PDR.
                if let Some(Ok(crate::ay_kinduction::KInductionResult::Counterexample { .. })) =
                    kinduction_result
                {
                    FusedWinner::KInduction
                } else if matches!(
                    bmc_result,
                    Some(Ok(crate::ay_bmc::BmcResult::Violation { .. }))
                ) {
                    FusedWinner::Bmc
                } else if matches!(pdr_result, Some(Ok(crate::ay_pdr::PdrResult::Unsafe { .. })))
                {
                    FusedWinner::Pdr
                } else {
                    FusedWinner::Bmc
                }
            }
        },
        _ => FusedWinner::Bfs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared_verdict::{SharedVerdict, Verdict};
    use crate::test_support::parse_module;
    use std::sync::Arc;

    #[test]
    fn test_fused_winner_variants_distinguishable() {
        let variants = [
            FusedWinner::Bfs,
            FusedWinner::Bmc,
            FusedWinner::Pdr,
            FusedWinner::KInduction,
        ];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j]);
            }
        }
    }

    // ---- #4 verdict-masking reconciliation (fail-closed) ----
    //
    // These tests pin the soundness contract of
    // `FusedResult::reconcile_masked_violation`: a symbolic (BMC) violation is
    // promoted to a CLI violation ONLY when the explicit-state evaluator
    // confirmed it; an unconfirmed symbolic claim must NOT override a clean BFS.
    #[cfg(feature = "ay")]
    use crate::CheckStats;

    #[cfg(feature = "ay")]
    fn mk_fused_result(
        winner: FusedWinner,
        bfs_result: CheckResult,
        cross_validation: Option<crate::check::cross_validation::CrossValidationResult>,
    ) -> FusedResult {
        FusedResult {
            winner,
            bfs_result,
            bmc_result: None,
            pdr_result: None,
            pdr_proof_replay_evidence: None,
            ay_shared_engine_evidence: Vec::new(),
            kinduction_result: None,
            symbolic_summary: None,
            degradation: SymbolicDegradation::default(),
            symbolic_coverage: 0.0,
            cross_validation,
        }
    }

    #[cfg(feature = "ay")]
    fn cv(engine_agrees: bool) -> crate::check::cross_validation::CrossValidationResult {
        use crate::check::cross_validation::CrossValidationSource;
        cv_for(engine_agrees, CrossValidationSource::Bmc, 3)
    }

    #[cfg(feature = "ay")]
    fn cv_for(
        engine_agrees: bool,
        source: crate::check::cross_validation::CrossValidationSource,
        trace_length: usize,
    ) -> crate::check::cross_validation::CrossValidationResult {
        // Confirmed counterexample rows (agrees + non-zero trace) carry the
        // replayed payload, exactly as `cross_validate_symbolic_trace` produces.
        let is_cex_confirmation = engine_agrees && trace_length > 0;
        crate::check::cross_validation::CrossValidationResult {
            engine_agrees,
            trace_length,
            source_engine: source,
            detail: format!(
                "{} counterexample replayed through BFS evaluator",
                source.lane_name()
            ),
            violated_invariant: is_cex_confirmation.then(|| "Inv".to_string()),
            validated_trace: is_cex_confirmation.then(|| {
                let mut assignments = std::collections::HashMap::new();
                assignments.insert("x".to_string(), tla_ay::BmcValue::Int(18));
                crate::check::cross_validation::bmc_states_to_trace(&[tla_ay::BmcState {
                    step: 0,
                    assignments,
                }])
                .expect("test trace converts")
            }),
        }
    }

    /// A one-state symbolic trace for cex-bearing lane results in tests.
    #[cfg(feature = "ay")]
    fn one_state_trace() -> Vec<tla_ay::BmcState> {
        let mut assignments = std::collections::HashMap::new();
        assignments.insert("x".to_string(), tla_ay::BmcValue::Int(18));
        vec![tla_ay::BmcState {
            step: 0,
            assignments,
        }]
    }

    // ---- k-Induction / PDR masked-violation reconciliation (the k-Induction
    // verdict-masking bug: a k-Ind base-case counterexample won the Violated
    // race, BFS was race-truncated into a clean "Success", and the CLI printed
    // "No error has been found. Resolved by: k-Induction" with exit 0). ----

    /// A CONFIRMED k-Induction base-case counterexample over a race-truncated
    /// clean BFS result MUST be promoted to a violation — the same treatment
    /// BMC violations already get.
    #[cfg(feature = "ay")]
    #[test]
    fn reconcile_promotes_confirmed_kinduction_base_case_cex() {
        use crate::check::cross_validation::CrossValidationSource;
        let mut r = mk_fused_result(
            FusedWinner::KInduction,
            CheckResult::Success(CheckStats::default()),
            Some(cv_for(true, CrossValidationSource::KInduction, 1)),
        );
        r.kinduction_result = Some(Ok(crate::ay_kinduction::KInductionResult::Counterexample {
            depth: 18,
            trace: one_state_trace(),
        }));
        match r.reconcile_masked_violation() {
            ReconciledVerdict::SymbolicViolation { lane, trace, .. } => {
                assert_eq!(lane, "k-Induction");
                assert!(!trace.is_empty(), "promoted verdict must carry the trace");
            }
            other => panic!("expected SymbolicViolation, got {other:?}"),
        }
    }

    /// A SPURIOUS (evaluator-rejected) k-Induction counterexample must NOT be
    /// promoted — and must NOT let the race-truncated BFS "Success" be reported
    /// as "no error" either: FAIL CLOSED to an inconclusive verdict.
    #[cfg(feature = "ay")]
    #[test]
    fn reconcile_fails_closed_on_spurious_kinduction_cex() {
        use crate::check::cross_validation::CrossValidationSource;
        let mut r = mk_fused_result(
            FusedWinner::KInduction,
            CheckResult::Success(CheckStats::default()),
            Some(cv_for(false, CrossValidationSource::KInduction, 1)),
        );
        r.kinduction_result = Some(Ok(crate::ay_kinduction::KInductionResult::Counterexample {
            depth: 18,
            trace: one_state_trace(),
        }));
        match r.reconcile_masked_violation() {
            ReconciledVerdict::UnvalidatedSymbolicViolation { lane, .. } => {
                assert_eq!(lane, "k-Induction")
            }
            other => panic!("expected fail-closed UnvalidatedSymbolicViolation, got {other:?}"),
        }
    }

    /// A LEGITIMATE k-Induction safety proof (Proved) that won the Satisfied
    /// race is not a masked violation — no over-correction: FromBfs.
    #[cfg(feature = "ay")]
    #[test]
    fn reconcile_leaves_kinduction_proved_win_untouched() {
        use crate::check::cross_validation::CrossValidationSource;
        let mut r = mk_fused_result(
            FusedWinner::KInduction,
            CheckResult::Success(CheckStats::default()),
            // Safety-proof cross-validation rows have trace_length 0.
            Some(cv_for(true, CrossValidationSource::KInduction, 0)),
        );
        r.kinduction_result = Some(Ok(crate::ay_kinduction::KInductionResult::Proved { k: 3 }));
        assert_eq!(r.reconcile_masked_violation(), ReconciledVerdict::FromBfs);
    }

    /// A CONFIRMED PDR unsafe trace over a race-truncated clean BFS result MUST
    /// be promoted — the PDR-equivalent of the k-Induction hole.
    #[cfg(feature = "ay")]
    #[test]
    fn reconcile_promotes_confirmed_pdr_unsafe_cex() {
        use crate::check::cross_validation::CrossValidationSource;
        let mut r = mk_fused_result(
            FusedWinner::Pdr,
            CheckResult::Success(CheckStats::default()),
            Some(cv_for(true, CrossValidationSource::Pdr, 1)),
        );
        let mut assignments = std::collections::HashMap::new();
        assignments.insert("x".to_string(), 18_i64);
        r.pdr_result = Some(Ok(crate::ay_pdr::PdrResult::Unsafe {
            trace: vec![tla_ay::chc::PdrState { assignments }],
        }));
        match r.reconcile_masked_violation() {
            ReconciledVerdict::SymbolicViolation { lane, trace, .. } => {
                assert_eq!(lane, "PDR");
                assert!(!trace.is_empty(), "promoted verdict must carry the trace");
            }
            other => panic!("expected SymbolicViolation, got {other:?}"),
        }
    }

    /// An UNCONFIRMED PDR unsafe race win fails closed to inconclusive.
    #[cfg(feature = "ay")]
    #[test]
    fn reconcile_fails_closed_on_unconfirmed_pdr_unsafe() {
        use crate::check::cross_validation::CrossValidationSource;
        let mut r = mk_fused_result(
            FusedWinner::Pdr,
            CheckResult::Success(CheckStats::default()),
            Some(cv_for(false, CrossValidationSource::Pdr, 1)),
        );
        let mut assignments = std::collections::HashMap::new();
        assignments.insert("x".to_string(), 18_i64);
        r.pdr_result = Some(Ok(crate::ay_pdr::PdrResult::Unsafe {
            trace: vec![tla_ay::chc::PdrState { assignments }],
        }));
        match r.reconcile_masked_violation() {
            ReconciledVerdict::UnvalidatedSymbolicViolation { lane, .. } => {
                assert_eq!(lane, "PDR")
            }
            other => panic!("expected fail-closed UnvalidatedSymbolicViolation, got {other:?}"),
        }
    }

    /// A LEGITIMATE PDR safety proof (Safe) that won the Satisfied race is not
    /// a masked violation — no over-correction: FromBfs.
    #[cfg(feature = "ay")]
    #[test]
    fn reconcile_leaves_pdr_safe_win_untouched() {
        use crate::check::cross_validation::CrossValidationSource;
        let mut r = mk_fused_result(
            FusedWinner::Pdr,
            CheckResult::Success(CheckStats::default()),
            Some(cv_for(true, CrossValidationSource::Pdr, 0)),
        );
        r.pdr_result = Some(Ok(crate::ay_pdr::PdrResult::Safe {
            invariant: "Inv".to_string(),
        }));
        assert_eq!(r.reconcile_masked_violation(), ReconciledVerdict::FromBfs);
    }

    /// A violation found by the explicit BFS lane is authoritative — the
    /// reconciliation never rewrites it, whatever the symbolic lanes claim.
    #[cfg(feature = "ay")]
    #[test]
    fn reconcile_never_overrides_bfs_violation() {
        let r = mk_fused_result(
            FusedWinner::Bmc,
            CheckResult::InvariantViolation {
                invariant: "Inv".to_string(),
                trace: crate::check::Trace::new(),
                stats: CheckStats::default(),
            },
            Some(cv(true)),
        );
        assert_eq!(r.reconcile_masked_violation(), ReconciledVerdict::FromBfs);
    }

    #[cfg(feature = "ay")]
    #[test]
    fn reconcile_promotes_confirmed_bmc_violation_over_clean_bfs() {
        // BFS finished clean, but BMC won (shared verdict Violated) AND the
        // explicit evaluator confirmed the counterexample → must report it.
        let r = mk_fused_result(
            FusedWinner::Bmc,
            CheckResult::Success(CheckStats::default()),
            Some(cv(true)),
        );
        match r.reconcile_masked_violation() {
            ReconciledVerdict::SymbolicViolation { lane, .. } => assert_eq!(lane, "BMC"),
            other => panic!("expected SymbolicViolation, got {other:?}"),
        }
    }

    #[cfg(feature = "ay")]
    #[test]
    fn reconcile_does_not_promote_unconfirmed_bmc_violation() {
        // BMC won the Violated race (truncating BFS) but the evaluator did NOT
        // confirm — never promote an unvalidated symbolic claim, and never
        // report the truncated BFS "Success" as "no error" either: FAIL CLOSED
        // to an inconclusive verdict.
        let r = mk_fused_result(
            FusedWinner::Bmc,
            CheckResult::Success(CheckStats::default()),
            Some(cv(false)),
        );
        match r.reconcile_masked_violation() {
            ReconciledVerdict::UnvalidatedSymbolicViolation { lane, .. } => {
                assert_eq!(lane, "BMC")
            }
            other => panic!("expected fail-closed UnvalidatedSymbolicViolation, got {other:?}"),
        }

        // Same, but cross-validation never ran at all (cex-bearing result lost:
        // e.g. the publishing lane panicked after publishing). Still fail closed.
        let r2 = mk_fused_result(
            FusedWinner::Bmc,
            CheckResult::Success(CheckStats::default()),
            None,
        );
        match r2.reconcile_masked_violation() {
            ReconciledVerdict::UnvalidatedSymbolicViolation { lane, .. } => {
                assert_eq!(lane, "BMC")
            }
            other => panic!("expected fail-closed UnvalidatedSymbolicViolation, got {other:?}"),
        }
    }

    #[cfg(feature = "ay")]
    #[test]
    fn reconcile_leaves_bfs_winner_untouched() {
        // BFS won — no reconciliation, even if a cross-validation row exists.
        let r = mk_fused_result(
            FusedWinner::Bfs,
            CheckResult::Success(CheckStats::default()),
            Some(cv(true)),
        );
        assert_eq!(r.reconcile_masked_violation(), ReconciledVerdict::FromBfs);
    }

    #[test]
    fn test_fused_result_first_verdict_populated() {
        // Verify that the non-ay path produces a coherent result.
        // (Full integration test requires a parsed TLA+ module.)
        let first_verdict = match FusedWinner::Bfs {
            FusedWinner::Bfs => "BFS",
            FusedWinner::Bmc => "BMC",
            FusedWinner::Pdr => "PDR",
            FusedWinner::KInduction => "k-Induction",
        };
        assert_eq!(first_verdict, "BFS");
    }

    #[test]
    fn test_shared_verdict_cooperative_race() {
        let sv = Arc::new(SharedVerdict::new());
        let sv1 = sv.clone();
        let sv2 = sv.clone();
        let sv3 = sv.clone();

        std::thread::scope(|scope| {
            // Lane 1: BFS publishes Satisfied.
            scope.spawn(move || {
                sv1.publish(Verdict::Satisfied);
            });

            // Lane 2: BMC checks and exits early.
            scope.spawn(move || {
                while !sv2.is_resolved() {
                    std::thread::yield_now();
                }
            });

            // Lane 3: PDR checks and exits early.
            scope.spawn(move || {
                while !sv3.is_resolved() {
                    std::thread::yield_now();
                }
            });
        });

        assert!(sv.is_resolved());
        assert_eq!(sv.get(), Some(Verdict::Satisfied));
    }

    #[test]
    fn test_three_lane_concurrent_race() {
        let sv = Arc::new(SharedVerdict::new());
        let handles: Vec<_> = (0..3)
            .map(|i| {
                let sv = sv.clone();
                std::thread::spawn(move || {
                    let v = match i {
                        0 => Verdict::Satisfied,
                        1 => Verdict::Violated,
                        _ => Verdict::Unknown,
                    };
                    sv.publish(v)
                })
            })
            .collect();

        let wins: Vec<bool> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        // At most one definitive publisher wins (Unknown doesn't count).
        let definitive_wins = wins[..2].iter().filter(|&&w| w).count();
        assert!(definitive_wins <= 1);
        // At least one definitive verdict should be published (Satisfied or Violated).
        // Unknown (lane 2) never resolves, so either lane 0 or lane 1 must win.
        assert!(sv.is_resolved());
    }

    // ========================================================================
    // Integration tests: FusedOrchestrator with real parsed TLA+ specs
    // ========================================================================

    /// Simple 2-state spec that passes all invariants.
    const PASSING_SPEC: &str = r#"
---- MODULE FusedPass ----
VARIABLE x
Init == x \in {0, 1}
Next == x' = 1 - x
Inv == x \in {0, 1}
====
"#;

    /// Spec whose invariant is violated: x eventually reaches 2.
    const VIOLATING_SPEC: &str = r#"
---- MODULE FusedViolate ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
Inv == x < 2
====
"#;

    /// Bounded counter that terminates (deadlocks) — no invariant.
    const DEADLOCK_FREE_SPEC: &str = r#"
---- MODULE FusedDeadlockFree ----
VARIABLE x
Init == x = 0
Next == x' = 1 - x
====
"#;

    #[test]
    fn test_fused_orchestrator_passing_spec() {
        let _lock = crate::test_utils::acquire_interner_lock();

        let module = parse_module(PASSING_SPEC);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Inv".to_string()],
            ..Default::default()
        };

        let orchestrator = FusedOrchestrator::new(&module, &[], &config);
        let result = orchestrator.run();

        // BFS always runs and should win in non-ay mode.
        assert_eq!(result.winner, FusedWinner::Bfs);
        // Non-ay: symbolic_summary is None.
        // ay: symbolic_summary is Some("Winner: ...").
        #[cfg(not(feature = "ay"))]
        assert!(result.symbolic_summary.is_none());
        #[cfg(feature = "ay")]
        assert!(result.symbolic_summary.is_some());

        // BFS should find exactly 2 states ({x=0, x=1}) and succeed.
        match &result.bfs_result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.states_found, 2, "expected 2 states for toggle spec");
            }
            other => panic!("Expected Success for passing spec, got: {other:?}"),
        }
    }

    #[test]
    fn test_fused_orchestrator_invariant_violation() {
        let _lock = crate::test_utils::acquire_interner_lock();

        let module = parse_module(VIOLATING_SPEC);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Inv".to_string()],
            ..Default::default()
        };

        let orchestrator = FusedOrchestrator::new(&module, &[], &config);
        let result = orchestrator.run();

        assert_eq!(result.winner, FusedWinner::Bfs);

        // BFS should detect invariant violation when x reaches 2.
        match &result.bfs_result {
            CheckResult::InvariantViolation { .. } => {}
            other => panic!("Expected InvariantViolation for violating spec, got: {other:?}"),
        }
    }

    #[test]
    fn test_fused_orchestrator_no_invariant_deadlock_free() {
        let _lock = crate::test_utils::acquire_interner_lock();

        let module = parse_module(DEADLOCK_FREE_SPEC);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec![],
            ..Default::default()
        };

        let orchestrator = FusedOrchestrator::new(&module, &[], &config);
        let result = orchestrator.run();

        assert_eq!(result.winner, FusedWinner::Bfs);

        // Toggle spec with no invariant: 2 states, no violation.
        match &result.bfs_result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.states_found, 2);
            }
            other => panic!("Expected Success for deadlock-free spec, got: {other:?}"),
        }
    }

    const DEADLOCK_REACHING_SPEC: &str = r#"
---- MODULE FusedDeadlockReaching ----
VARIABLE x
Init == x = 0
Next == x = 0 /\ x' = 1
====
"#;

    // A spec whose invariant (x <= 5000) is inductive — so PDR/k-induction can prove
    // SAFETY and publish Verdict::Satisfied — but which REACHES A DEADLOCK at x = 5000
    // (Next disabled), DEEPER than the 4096-state portfolio poll interval. Without the
    // soundness gate, a symbolic Satisfied could win the cooperative race and truncate
    // the BFS lane (which polls is_resolved every 4096 states) into an indistinguishable
    // Success, silently masking the reachable deadlock. Used by
    // test_fused_symbolic_safe_does_not_mask_deadlock.
    const SAFE_INVARIANT_WITH_DEEP_DEADLOCK_SPEC: &str = r#"
---- MODULE FusedSafeDeepDeadlock ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Next == x < 5000 /\ x' = x + 1
Inv == x <= 5000
====
"#;

    // Multi-variable counting spec whose invariant `a < 3` is violated when `a`
    // reaches 3. BFS finds the violation in milliseconds, but the PDR lane's
    // CHC fixpoint on this 6-variable / 4-disjunct system does not converge
    // quickly — so the lane used to keep solving up to its 300s `solve_timeout`,
    // hanging the whole fused run. See `solve_pdr_interruptible`.
    const FUSED_PDR_SLOW_VIOLATION_SPEC: &str = r#"
---- MODULE FusedPdrSlowViolation ----
EXTENDS Naturals
VARIABLES a, b, c, d, e, f
Init == a = 0 /\ b = 0 /\ c = 0 /\ d = 0 /\ e = 0 /\ f = 0
StepAB == a < 3 /\ a' = a + 1 /\ b' = b + 1 /\ UNCHANGED <<c, d, e, f>>
StepC  == a >= 1 /\ c < 3 /\ c' = c + 1 /\ UNCHANGED <<a, b, d, e, f>>
StepDE == c >= 1 /\ d < 3 /\ d' = d + 1 /\ e' = e + 1 /\ UNCHANGED <<a, b, c, f>>
StepF  == d >= 1 /\ f < 3 /\ f' = f + 1 /\ UNCHANGED <<a, b, c, d, e>>
Next == StepAB \/ StepC \/ StepDE \/ StepF
SafeInvariant == a < 3
====
"#;

    /// Regression for the fused-mode deadlock HANG. A spec that REACHES a
    /// terminal (no-successor) state under default deadlock-checking used to spin
    /// the wavefront-compressor thread forever: `CheckResult::Deadlock` publishes
    /// no racing verdict (it maps to `Verdict::Unknown`, a no-op), so
    /// `is_resolved()` never became true and the `thread::scope` join blocked.
    /// The orchestrator must now TERMINATE and report the deadlock. If this test
    /// ever hangs, the `is_bfs_complete()` exit in the wavefront loop regressed.
    #[test]
    fn test_fused_orchestrator_reaches_deadlock() {
        let _lock = crate::test_utils::acquire_interner_lock();

        let module = parse_module(DEADLOCK_REACHING_SPEC);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec![],
            // check_deadlock defaults to true (TLC default), so x=1 (terminal) is
            // a reported deadlock rather than a silently-accepted terminal state.
            ..Default::default()
        };

        let orchestrator = FusedOrchestrator::new(&module, &[], &config);
        let result = orchestrator.run();

        match &result.bfs_result {
            CheckResult::Deadlock { stats, .. } => {
                // x=0 (init) and x=1 (terminal) reached before the deadlock.
                assert_eq!(stats.states_found, 2);
            }
            other => panic!("Expected Deadlock for deadlock-reaching spec, got: {other:?}"),
        }
    }

    /// SOUNDNESS regression — symbolic-safe verdict masking (the symmetric dual of the
    /// #4 violation-masking fix). PDR and k-Induction prove the SAFETY invariant ONLY;
    /// they do not verify deadlock-freedom. With deadlock-checking on (the TLC-parity
    /// default), a symbolic `Verdict::Satisfied` must NOT resolve the cooperative
    /// verdict — doing so makes the BFS lane exit early and return a `CheckResult`
    /// INDISTINGUISHABLE from a complete `Success`, silently masking a reachable
    /// deadlock. The symbolic safety lanes are gated by
    /// `ay_shared::symbolic_safety_proof_covers_all_obligations`, leaving BFS
    /// authoritative; this spec deadlocks DEEPER than the 4096-state portfolio poll so a
    /// pre-fix symbolic-safe race-win would have truncated BFS before x=5000. With the
    /// gate the run is deterministic: BFS reaches x=5000 and reports the Deadlock.
    #[cfg(feature = "ay")]
    #[test]
    fn test_fused_symbolic_safe_does_not_mask_deadlock() {
        let _lock = crate::test_utils::acquire_interner_lock();

        let module = parse_module(SAFE_INVARIANT_WITH_DEEP_DEADLOCK_SPEC);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            // PDR/k-induction can prove this invariant (x <= 5000 is inductive) and
            // would publish Satisfied; the gate must stop that from masking the deadlock.
            invariants: vec!["Inv".to_string()],
            // check_deadlock defaults to true (TLC parity).
            ..Default::default()
        };

        let orchestrator = FusedOrchestrator::new(&module, &[], &config);
        let result = orchestrator.run();

        assert!(
            matches!(result.bfs_result, CheckResult::Deadlock { .. }),
            "a safety-provable spec with a reachable deadlock must report Deadlock, not a \
             symbolic-safe-masked Success; got: {:?}",
            result.bfs_result
        );
    }

    /// Regression for the EARLY-RETURN variant of the fused-mode hang. When
    /// `check_impl` returns before reaching a BFS loop (here: an invariant that
    /// fails to compile), the BFS-loop finalizers never run, so `mark_bfs_complete`
    /// was never called and — in fused mode — the cooperative BMC/wavefront lanes
    /// spun forever. The orchestrator must now TERMINATE (with a non-Success
    /// result) rather than hang. The fix is the unconditional `mark_bfs_complete()`
    /// at the end of `ModelChecker::check()`.
    #[test]
    fn test_fused_orchestrator_setup_error_terminates() {
        let _lock = crate::test_utils::acquire_interner_lock();

        let module = parse_module(DEADLOCK_FREE_SPEC);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            // A non-existent invariant makes check_impl fail during setup and
            // early-return before any BFS loop runs.
            invariants: vec!["NoSuchInvariant".to_string()],
            ..Default::default()
        };

        let orchestrator = FusedOrchestrator::new(&module, &[], &config);
        let result = orchestrator.run();

        // The point is that run() RETURNS at all (pre-fix it hung). A
        // failed-to-compile invariant must not masquerade as Success.
        assert!(
            !matches!(result.bfs_result, CheckResult::Success(_)),
            "expected a non-Success terminal result for a bad-invariant spec, got: {:?}",
            result.bfs_result
        );
    }

    /// Regression for the PDR-lane fused HANG. On a spec where BFS finds the
    /// violation immediately but the PDR CHC fixpoint does not converge, the PDR
    /// lane's *blocking* `solve_pdr_with_proof_evidence` ignored the verdict
    /// another lane had already published — it only checked `is_resolved()`
    /// before/after the solve, never during — so the `thread::scope` join blocked
    /// up to the 300s `solve_timeout`. The lane must now tear down promptly once
    /// the verdict is resolved (`solve_pdr_interruptible`). If this test ever
    /// hangs, that interruptibility regressed.
    #[cfg(feature = "ay")]
    #[test]
    fn test_fused_orchestrator_pdr_lane_does_not_hang_on_violation() {
        let _lock = crate::test_utils::acquire_interner_lock();

        let module = parse_module(FUSED_PDR_SLOW_VIOLATION_SPEC);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["SafeInvariant".to_string()],
            // Match the spec's intent: this is an invariant violation, not a
            // deadlock check (the spec deliberately reaches terminal states).
            check_deadlock: false,
            ..Default::default()
        };

        let orchestrator = FusedOrchestrator::new(&module, &[], &config);
        let result = orchestrator.run();

        // BFS must report the `a < 3` violation — and crucially, run() must
        // RETURN (pre-fix the PDR lane hung the scope join). The exact winner is
        // not asserted; termination + a violation verdict is the contract.
        match &result.bfs_result {
            CheckResult::InvariantViolation { invariant, .. } => {
                assert_eq!(invariant, "SafeInvariant");
            }
            other => panic!("Expected InvariantViolation, got: {other:?}"),
        }
    }

    #[cfg(feature = "ay")]
    #[test]
    fn test_fused_orchestrator_ay_four_lane_passing() {
        let _lock = crate::test_utils::acquire_interner_lock();

        let module = parse_module(PASSING_SPEC);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Inv".to_string()],
            ..Default::default()
        };

        let orchestrator = FusedOrchestrator::new(&module, &[], &config);
        let result = orchestrator.run();

        // With ay: all four lanes run. BFS should still find 2 states.
        match &result.bfs_result {
            CheckResult::Success(stats) => {
                assert_eq!(stats.states_found, 2);
            }
            other => panic!("Expected Success from BFS lane, got: {other:?}"),
        }

        // BMC, PDR, and k-Induction results should be present (may be Ok or Err
        // depending on ay translation support for this simple spec).
        assert!(
            result.bmc_result.is_some(),
            "BMC result should be present with ay feature"
        );
        assert!(
            result.pdr_result.is_some(),
            "PDR result should be present with ay feature"
        );
        let pdr_evidence = result
            .pdr_proof_replay_evidence()
            .expect("PDR proof/replay evidence should be visible on fused result");
        assert!(pdr_evidence.contains("TLA ay_chc_proof_replay_boundary"));
        assert!(pdr_evidence.contains("production_selected=false"));
        assert!(pdr_evidence.contains("fail_closed=true"));
        assert!(result.ay_shared_engine_evidence().iter().any(|row| {
            row.contains("TLA ay_shared_engine_metadata")
                && row.contains("lanes=all_sat_enumeration,bmc,chc,pdr,k_induction")
        }));
        assert!(
            result.kinduction_result.is_some(),
            "k-Induction result should be present with ay feature"
        );
    }

    #[cfg(feature = "ay")]
    #[test]
    fn test_determine_fused_winner_bfs_satisfied() {
        let sv = SharedVerdict::new();
        sv.publish(Verdict::Satisfied);
        let bfs_result = CheckResult::Success(crate::check::CheckStats {
            states_found: 2,

            ..Default::default()
        });
        assert_eq!(
            determine_fused_winner(&sv, &bfs_result, &None, &None, &None),
            FusedWinner::Bfs,
        );
    }

    #[cfg(feature = "ay")]
    #[test]
    fn test_determine_fused_winner_pdr_satisfied() {
        let sv = SharedVerdict::new();
        sv.publish(Verdict::Satisfied);
        // BFS did not reach success (e.g., still running) — PDR resolved first.
        let bfs_result = CheckResult::LimitReached {
            limit_type: crate::check::LimitType::States,
            stats: crate::check::CheckStats::default(),
        };
        assert_eq!(
            determine_fused_winner(&sv, &bfs_result, &None, &None, &None),
            FusedWinner::Pdr,
        );
    }

    #[cfg(feature = "ay")]
    #[test]
    fn test_determine_fused_winner_bmc_violated() {
        let sv = SharedVerdict::new();
        sv.publish(Verdict::Violated);
        // BFS did not find the violation — BMC found it.
        let bfs_result = CheckResult::Success(crate::check::CheckStats::default());
        let bmc_result = Some(Ok(crate::ay_bmc::BmcResult::Violation {
            depth: 2,
            trace: vec![],
        }));
        assert_eq!(
            determine_fused_winner(&sv, &bfs_result, &bmc_result, &None, &None),
            FusedWinner::Bmc,
        );
        // Fallback attribution: verdict Violated but no lane result carries the
        // counterexample (lost/panicked lane) — still Bmc, so the reconciliation
        // can fail closed on the missing cross-validation.
        assert_eq!(
            determine_fused_winner(&sv, &bfs_result, &None, &None, &None),
            FusedWinner::Bmc,
        );
    }

    /// PDR unsafe traces attribute the Violated win to PDR (previously
    /// misattributed to BMC, so the masked-violation reconciliation could not
    /// pair the winner with the PDR counterexample).
    #[cfg(feature = "ay")]
    #[test]
    fn test_determine_fused_winner_pdr_unsafe_violated() {
        let sv = SharedVerdict::new();
        sv.publish(Verdict::Violated);
        let bfs_result = CheckResult::Success(crate::check::CheckStats::default());
        let pdr_result = Some(Ok(crate::ay_pdr::PdrResult::Unsafe { trace: vec![] }));
        assert_eq!(
            determine_fused_winner(&sv, &bfs_result, &None, &pdr_result, &None),
            FusedWinner::Pdr,
        );
    }

    /// Part of #3844: k-Induction proving safety attributes the win correctly.
    #[cfg(feature = "ay")]
    #[test]
    fn test_determine_fused_winner_kinduction_proved() {
        let sv = SharedVerdict::new();
        sv.publish(Verdict::Satisfied);
        // BFS did not reach success — k-Induction proved safety first.
        let bfs_result = CheckResult::LimitReached {
            limit_type: crate::check::LimitType::States,
            stats: crate::check::CheckStats::default(),
        };
        let kind_result = Some(Ok(crate::ay_kinduction::KInductionResult::Proved { k: 3 }));
        assert_eq!(
            determine_fused_winner(&sv, &bfs_result, &None, &None, &kind_result),
            FusedWinner::KInduction,
        );
    }

    /// Part of #3844: k-Induction base-case counterexample attributes violation.
    #[cfg(feature = "ay")]
    #[test]
    fn test_determine_fused_winner_kinduction_counterexample() {
        let sv = SharedVerdict::new();
        sv.publish(Verdict::Violated);
        // BFS did not find the violation — k-Induction base case found it.
        let bfs_result = CheckResult::Success(crate::check::CheckStats::default());
        let kind_result = Some(Ok(crate::ay_kinduction::KInductionResult::Counterexample {
            depth: 2,
            trace: vec![],
        }));
        assert_eq!(
            determine_fused_winner(&sv, &bfs_result, &None, &None, &kind_result),
            FusedWinner::KInduction,
        );
    }

    // ---- Part of #3837: SymbolicDegradation tests ----

    #[test]
    fn test_degradation_default_no_degradation() {
        let d = SymbolicDegradation::default();
        assert!(!d.any_degraded());
        assert!((d.symbolic_coverage() - 1.0).abs() < f64::EPSILON);
        assert!(d.summary().is_none());
    }

    #[test]
    fn test_degradation_bmc_only() {
        let d = SymbolicDegradation {
            bmc_degraded: true,
            bmc_reason: Some("translation failed: unsupported operator CHOOSE".to_string()),
            unsupported_constructs: vec!["CHOOSE".to_string()],
            ..Default::default()
        };
        assert!(d.any_degraded());
        // 1 of 3 symbolic lanes degraded => 2/3 lane coverage.
        assert!((d.lane_coverage() - 2.0 / 3.0).abs() < 0.01);
        let summary = d.summary().unwrap();
        assert!(summary.contains("66%"));
        assert!(summary.contains("BMC"));
        assert!(summary.contains("CHOOSE"));
    }

    #[test]
    fn test_degradation_bmc_and_pdr_lanes() {
        let d = SymbolicDegradation {
            bmc_degraded: true,
            bmc_reason: Some("solver failed".to_string()),
            pdr_degraded: true,
            pdr_reason: Some("sort inference failed".to_string()),
            ..Default::default()
        };
        assert!(d.any_degraded());
        // 2 of 3 symbolic lanes degraded => 1/3 lane coverage = 33%.
        assert!((d.lane_coverage() - 1.0 / 3.0).abs() < 0.01);
        let summary = d.summary().unwrap();
        assert!(summary.contains("33%"));
        assert!(summary.contains("BMC"));
        assert!(summary.contains("PDR"));
    }

    #[test]
    fn test_degradation_all_symbolic_lanes() {
        let d = SymbolicDegradation {
            bmc_degraded: true,
            bmc_reason: Some("solver failed".to_string()),
            pdr_degraded: true,
            pdr_reason: Some("sort inference failed".to_string()),
            kinduction_degraded: true,
            kinduction_reason: Some("translation failed".to_string()),
            ..Default::default()
        };
        assert!(d.any_degraded());
        assert!(d.lane_coverage().abs() < f64::EPSILON);
        let summary = d.summary().unwrap();
        assert!(summary.contains("0%"));
        assert!(summary.contains("BMC"));
        assert!(summary.contains("PDR"));
        assert!(summary.contains("k-Induction"));
    }

    #[cfg(feature = "ay")]
    #[test]
    fn test_extract_unsupported_constructs_from_message() {
        let constructs =
            extract_unsupported_constructs("unsupported operator: CHOOSE in SetFilter context");
        assert!(constructs.contains(&"CHOOSE".to_string()));
        assert!(constructs.contains(&"SetFilter".to_string()));
    }

    #[cfg(feature = "ay")]
    #[test]
    fn test_build_degradation_both_ok() {
        let bmc = Some(Ok(crate::ay_bmc::BmcResult::BoundReached { max_depth: 5 }));
        let pdr = Some(Ok(crate::ay_pdr::PdrResult::Safe {
            invariant: "Inv".to_string(),
        }));
        let kind = Some(Ok(crate::ay_kinduction::KInductionResult::Unknown {
            max_k: 5,
            reason: "inconclusive".to_string(),
        }));
        let flags = vec![true, true];
        let names = vec!["A".to_string(), "B".to_string()];
        let deg = build_degradation(&bmc, &pdr, &kind, &flags, &names);
        assert!(!deg.any_degraded());
        assert!((deg.lane_coverage() - 1.0).abs() < f64::EPSILON);
    }

    #[cfg(feature = "ay")]
    #[test]
    fn test_build_degradation_bmc_error() {
        let bmc: Option<Result<crate::ay_bmc::BmcResult, crate::ay_bmc::BmcError>> = Some(Err(
            crate::ay_bmc::BmcError::TranslationError("unsupported operator: CHOOSE".to_string()),
        ));
        let pdr = Some(Ok(crate::ay_pdr::PdrResult::Safe {
            invariant: "Inv".to_string(),
        }));
        let kind: Option<
            Result<crate::ay_kinduction::KInductionResult, crate::ay_kinduction::KInductionError>,
        > = Some(Ok(crate::ay_kinduction::KInductionResult::Unknown {
            max_k: 5,
            reason: "inconclusive".to_string(),
        }));
        let flags = vec![true, false];
        let names = vec!["A".to_string(), "B".to_string()];
        let deg = build_degradation(&bmc, &pdr, &kind, &flags, &names);
        assert!(deg.bmc_degraded);
        assert!(!deg.pdr_degraded);
        assert!(!deg.kinduction_degraded);
        // 1 of 3 symbolic lanes degraded => 2/3 lane coverage.
        assert!((deg.lane_coverage() - 2.0 / 3.0).abs() < 0.01);
        assert!(deg.unsupported_constructs.contains(&"CHOOSE".to_string()));
        assert_eq!(deg.actions_total, 2);
        assert_eq!(deg.actions_smt_compatible, 1);
        assert!(deg.bmc_error.is_some());
    }

    #[cfg(feature = "ay")]
    #[test]
    fn test_build_degradation_all_panicked() {
        let bmc: Option<Result<crate::ay_bmc::BmcResult, crate::ay_bmc::BmcError>> = None;
        let pdr: Option<Result<crate::ay_pdr::PdrResult, crate::ay_pdr::PdrError>> = None;
        let kind: Option<
            Result<crate::ay_kinduction::KInductionResult, crate::ay_kinduction::KInductionError>,
        > = None;
        let flags: Vec<bool> = vec![];
        let names: Vec<String> = vec![];
        let deg = build_degradation(&bmc, &pdr, &kind, &flags, &names);
        assert!(deg.bmc_degraded);
        assert!(deg.pdr_degraded);
        assert!(deg.kinduction_degraded);
        // All 3 symbolic lanes degraded => 0% lane coverage.
        assert!(deg.lane_coverage().abs() < f64::EPSILON);
    }

    /// Part of #3844: k-Induction error tracked in degradation.
    #[cfg(feature = "ay")]
    #[test]
    fn test_build_degradation_kinduction_error() {
        let bmc = Some(Ok(crate::ay_bmc::BmcResult::BoundReached { max_depth: 5 }));
        let pdr = Some(Ok(crate::ay_pdr::PdrResult::Safe {
            invariant: "Inv".to_string(),
        }));
        let kind: Option<
            Result<crate::ay_kinduction::KInductionResult, crate::ay_kinduction::KInductionError>,
        > = Some(Err(
            crate::ay_kinduction::KInductionError::TranslationError(
                "unsupported operator: SUBSET".to_string(),
            ),
        ));
        let flags = vec![true];
        let names = vec!["A".to_string()];
        let deg = build_degradation(&bmc, &pdr, &kind, &flags, &names);
        assert!(!deg.bmc_degraded);
        assert!(!deg.pdr_degraded);
        assert!(deg.kinduction_degraded);
        assert!(deg
            .kinduction_reason
            .as_ref()
            .unwrap()
            .contains("translation failed"));
        assert!(deg.unsupported_constructs.contains(&"SUBSET".to_string()));
        // 1 of 3 symbolic lanes degraded => 2/3 lane coverage.
        assert!((deg.lane_coverage() - 2.0 / 3.0).abs() < 0.01);
    }

    /// CDEMC demo: 3 independent bounded counters in a record.
    /// BFS state space: 101^3 ~ 1M states. With max_states=1000, BFS hits
    /// the limit. PDR should prove safety symbolically (or degrade gracefully).
    ///
    /// Part of #3957.
    #[cfg(feature = "ay")]
    #[test]
    fn test_fused_counter_array_3_symbolic_advantage() {
        let _lock = crate::test_utils::acquire_interner_lock();

        let src = r#"
---- MODULE FusedCounterArray3 ----
EXTENDS Integers
VARIABLE c
Init == c = [c1 |-> 0, c2 |-> 0, c3 |-> 0]
Next == \/ c' = [c EXCEPT !.c1 = IF @ < 100 THEN @ + 1 ELSE @]
        \/ c' = [c EXCEPT !.c2 = IF @ < 100 THEN @ + 1 ELSE @]
        \/ c' = [c EXCEPT !.c3 = IF @ < 100 THEN @ + 1 ELSE @]
        \/ UNCHANGED c
Safety == /\ c.c1 >= 0 /\ c.c1 <= 100
          /\ c.c2 >= 0 /\ c.c2 <= 100
          /\ c.c3 >= 0 /\ c.c3 <= 100
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Safety".to_string()],
            ..Default::default()
        };

        let orchestrator = FusedOrchestrator::new(&module, &[], &config).with_checker_config(
            FusedCheckerConfig {
                max_states: 1000,
                ..Default::default()
            },
        );
        let result = orchestrator.run();

        // The spec is safe — regardless of which lane wins, there must be no
        // invariant violation.
        match &result.bfs_result {
            CheckResult::InvariantViolation { .. }
            | CheckResult::PropertyViolation { .. }
            | CheckResult::LivenessViolation { .. } => {
                panic!(
                    "CounterArray3 is safe — fused mode should not report violation: {:?}",
                    result.bfs_result
                );
            }
            _ => {
                // BFS hit limit, success, or PDR proved safety — all acceptable.
            }
        }

        // Verify all symbolic lanes attempted (may succeed or degrade).
        assert!(result.bmc_result.is_some(), "BMC lane should have run");
        assert!(result.pdr_result.is_some(), "PDR lane should have run");
        assert!(
            result.kinduction_result.is_some(),
            "k-Induction lane should have run"
        );

        // If PDR or k-Induction proved safety, the winner should reflect that.
        if matches!(result.winner, FusedWinner::Pdr | FusedWinner::KInduction) {
            eprintln!(
                "CDEMC demo success: symbolic engine ({:?}) proved safety for \
                 101^3 state space that BFS cannot enumerate",
                result.winner
            );
        }
    }

    /// SOUNDNESS regression — the k-Induction verdict-masking bug, end to end.
    ///
    /// A branching-8 spec with a REAL invariant violation at depth 18: the BFS
    /// lane cannot reach it (8^18 states), so a symbolic lane (k-Induction's
    /// BMC base case, or the cooperative BMC lane) finds it, publishes
    /// `Violated`, and race-truncates BFS into a clean-looking result. Before
    /// the fix, `reconcile_masked_violation` only promoted BMC violations —
    /// when k-Induction's base case was the finder, the CLI printed "No error
    /// has been found. Resolved by: k-Induction" with exit 0: a real violation
    /// masked as success.
    ///
    /// The pinned contract: for this spec the reconciliation NEVER yields
    /// `FromBfs` over a clean BFS result. The winning lane's counterexample is
    /// real (d = 18 with Inv == d < 18 is interpreter-confirmable), so the
    /// expected outcome is a full `SymbolicViolation` promotion, whichever
    /// symbolic lane wins the race.
    #[cfg(feature = "ay")]
    #[test]
    fn test_fused_kinduction_base_cex_not_masked_end_to_end() {
        let _lock = crate::test_utils::acquire_interner_lock();

        let src = r#"
---- MODULE FusedKindDeepViolation ----
EXTENDS Naturals
VARIABLES d, j
Init == d = 0 /\ j = 0
Next == \E c \in 0..7 : d' = d + 1 /\ j' = (j * 8) + c
Inv == d < 18
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Inv".to_string()],
            ..Default::default()
        };

        let orchestrator = FusedOrchestrator::new(&module, &[], &config);
        let result = orchestrator.run();

        match result.reconcile_masked_violation() {
            ReconciledVerdict::SymbolicViolation { lane, trace, .. } => {
                eprintln!(
                    "masked violation promoted from the {lane} lane ({} trace states)",
                    trace.len()
                );
            }
            ReconciledVerdict::UnvalidatedSymbolicViolation { lane, detail } => {
                panic!(
                    "the {lane} lane's counterexample is real (d = 18) and must \
                     cross-validate; fail-closed inconclusive means the validation \
                     regressed: {detail}"
                );
            }
            ReconciledVerdict::FromBfs => {
                // Only acceptable if BFS itself reported the violation (it cannot
                // finish this state space cleanly).
                assert!(
                    matches!(
                        result.bfs_result,
                        CheckResult::InvariantViolation { .. }
                            | CheckResult::PropertyViolation { .. }
                    ),
                    "k-Induction verdict-masking regressed: a real depth-18 violation \
                     was reported as the race-truncated BFS result {:?} (winner {:?})",
                    result.bfs_result,
                    result.winner
                );
            }
        }
    }
}
