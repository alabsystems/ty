// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Concrete `PhaseRunner` implementations that wire verification backends
//! into the multi-phase pipeline (`pipeline.rs`).
//!
//! Part of #3720, #3723.

use rustc_hash::FxHashMap;
use std::time::Duration;

use super::model_checker::random_walk::{RandomWalkConfig, RandomWalkResult};
use super::model_checker::ModelChecker;
use super::pipeline::{PhaseRunner, PropertyId, PropertyVerdict, VerificationPhase};

// ---------------------------------------------------------------------------
// RandomWalkRunner
// ---------------------------------------------------------------------------

/// Pipeline runner that wraps the random-walk witness search engine.
///
/// Runs `num_walks` independent random walks of up to `max_depth` steps each.
/// Any invariant violation found is a real witness (sound under-approximation);
/// properties that survive all walks are reported as `Unknown` so subsequent
/// pipeline phases can attempt proof or deeper exploration.
///
/// Part of #3720.
pub struct RandomWalkRunner<'a> {
    checker: ModelChecker<'a>,
    walk_config: RandomWalkConfig,
    deadlock_reached: bool,
}

impl std::fmt::Debug for RandomWalkRunner<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RandomWalkRunner")
            .field("walk_config", &self.walk_config)
            .finish()
    }
}

impl<'a> RandomWalkRunner<'a> {
    /// Create a new runner.
    ///
    /// `checker` is a freshly-constructed `ModelChecker` (the runner takes
    /// ownership because `random_walk` requires `&mut self`).
    /// `walk_config` controls number of walks, depth, and seed.
    pub fn new(checker: ModelChecker<'a>, walk_config: RandomWalkConfig) -> Self {
        Self {
            checker,
            walk_config,
            deadlock_reached: false,
        }
    }
}

impl PhaseRunner for RandomWalkRunner<'_> {
    fn run(
        &mut self,
        unresolved: &[PropertyId],
        _time_budget: Duration,
    ) -> FxHashMap<PropertyId, PropertyVerdict> {
        let result = self.checker.random_walk(&self.walk_config);

        let mut verdicts = FxHashMap::default();

        match result {
            RandomWalkResult::InvariantViolation { ref invariant, .. } => {
                // Mark the violated invariant as Violated.
                for prop in unresolved {
                    if prop == invariant {
                        verdicts.insert(prop.clone(), PropertyVerdict::Violated);
                    }
                    // Other properties remain unresolved (not in map).
                }
            }
            RandomWalkResult::Deadlock { .. } => {
                // Deadlock is a global property failure, not specific to a
                // single invariant — leave the per-invariant verdicts Unknown
                // but record the deadlock so the pipeline/CLI reports it
                // (`deadlock_reached`), instead of silently exiting 0.
                self.deadlock_reached = true;
            }
            RandomWalkResult::NoViolationFound { .. } => {
                // No violation found — all properties remain Unknown.
            }
            RandomWalkResult::Error(_) => {
                // Phase error — all properties remain Unknown so later
                // phases can still attempt verification.
            }
        }

        verdicts
    }

    fn phase(&self) -> VerificationPhase {
        VerificationPhase::RandomWalk
    }

    fn deadlock_reached(&self) -> bool {
        self.deadlock_reached
    }
}

// ---------------------------------------------------------------------------
// BFS Runner (always available)
// ---------------------------------------------------------------------------

/// BFS runner wrapping the explicit-state model checker for exhaustive
/// state-space exploration.
///
/// Maps:
/// - `CheckResult::Success` -> `PropertyVerdict::Satisfied` for all properties
/// - `CheckResult::InvariantViolation` -> `PropertyVerdict::Violated` for the
///   specific invariant that failed; other properties remain `Unknown`
/// - Other results (error, limit, deadlock) -> leaves as `Unknown`
///
/// Part of #3723.
pub struct BfsRunner<'a> {
    module: &'a tla_core::ast::Module,
    checker_modules: Vec<&'a tla_core::ast::Module>,
    config: &'a crate::config::Config,
    deadlock_reached: bool,
}

impl<'a> BfsRunner<'a> {
    /// Construct a BFS runner from the module and config.
    pub fn new(
        module: &'a tla_core::ast::Module,
        checker_modules: &[&'a tla_core::ast::Module],
        config: &'a crate::config::Config,
    ) -> Self {
        Self {
            module,
            checker_modules: checker_modules.to_vec(),
            config,
            deadlock_reached: false,
        }
    }
}

impl std::fmt::Debug for BfsRunner<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BfsRunner")
            .field("module", &self.module.name.node)
            .finish()
    }
}

impl PhaseRunner for BfsRunner<'_> {
    fn run(
        &mut self,
        unresolved: &[PropertyId],
        time_budget: Duration,
    ) -> FxHashMap<PropertyId, PropertyVerdict> {
        use super::api::CheckResult;

        let runtime_config = self.config.runtime_model_config();
        let mut checker =
            ModelChecker::new_with_extends(self.module, &self.checker_modules, &runtime_config);
        // Part of #3 (overrun/OOM time backstop): apply the phase's time budget
        // as a wall-clock deadline so an unbounded / explosive spec cannot make
        // BFS run forever or grow the state space until the process OOMs. When
        // the deadline elapses, `check()` returns `LimitReached { Time }`, which
        // the match below leaves as Unknown for every property — fail-closed.
        // A zero/non-positive budget means "no backstop" (set_time_budget
        // ignores a saturating-overflow Instant), preserving prior behavior for
        // callers that pass Duration::MAX or 0.
        if !time_budget.is_zero() {
            checker.set_time_budget(time_budget);
        }
        let result = checker.check();

        let mut verdicts = FxHashMap::default();
        match result {
            CheckResult::Success(_) => {
                // All invariants hold across the full state space.
                for prop in unresolved {
                    verdicts.insert(prop.clone(), PropertyVerdict::Satisfied);
                }
            }
            CheckResult::InvariantViolation { ref invariant, .. } => {
                // The specific invariant that failed.
                for prop in unresolved {
                    if prop == invariant {
                        verdicts.insert(prop.clone(), PropertyVerdict::Violated);
                    }
                    // Other properties remain unknown — BFS found one violation
                    // but we don't know about the rest.
                }
            }
            CheckResult::Deadlock { .. } => {
                // A reachable deadlock is a global property failure that does
                // not invalidate any single named invariant — leave per-property
                // verdicts Unknown but record the deadlock so the pipeline/CLI
                // reports it instead of silently exiting 0.
                self.deadlock_reached = true;
            }
            CheckResult::Vacuous { .. }
            | CheckResult::Error { .. }
            | CheckResult::LimitReached { .. }
            | CheckResult::PropertyViolation { .. }
            | CheckResult::LivenessViolation { .. } => {
                // Could not determine any property status. A vacuous run
                // proved nothing about the properties, so they stay unknown.
            }
        }
        verdicts
    }

    fn phase(&self) -> VerificationPhase {
        VerificationPhase::Bfs
    }

    fn deadlock_reached(&self) -> bool {
        self.deadlock_reached
    }
}

// ---------------------------------------------------------------------------
// BMC Runner (ay feature gate)
// ---------------------------------------------------------------------------

/// BMC runner wrapping `check_bmc()` for symbolic bounded bug finding.
///
/// Maps:
/// - `BmcResult::Violation` -> `PropertyVerdict::Violated` for all unresolved
///   properties (BMC checks the conjunction of all invariants)
/// - `BmcResult::BoundReached` -> leaves as `Unknown` (BMC cannot prove safety)
/// - `BmcResult::Unknown` -> leaves as `Unknown`
///
/// Part of #3723.
#[cfg(feature = "ay")]
pub struct BmcRunner<'a> {
    module: &'a tla_core::ast::Module,
    config: &'a crate::config::Config,
    ctx: &'a crate::eval::EvalCtx,
    /// BMC depth bound (pipeline default: 20).
    default_depth: usize,
}

#[cfg(feature = "ay")]
impl<'a> BmcRunner<'a> {
    /// Construct a BMC runner.
    ///
    /// `default_depth` controls the BMC unrolling depth. A reasonable default
    /// is 10-30 for pipeline use (deeper exploration happens in later phases).
    pub fn new(
        module: &'a tla_core::ast::Module,
        config: &'a crate::config::Config,
        ctx: &'a crate::eval::EvalCtx,
        default_depth: usize,
    ) -> Self {
        Self {
            module,
            config,
            ctx,
            default_depth,
        }
    }
}

#[cfg(feature = "ay")]
impl std::fmt::Debug for BmcRunner<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BmcRunner")
            .field("module", &self.module.name.node)
            .field("default_depth", &self.default_depth)
            .finish()
    }
}

#[cfg(feature = "ay")]
impl PhaseRunner for BmcRunner<'_> {
    fn run(
        &mut self,
        unresolved: &[PropertyId],
        time_budget: Duration,
    ) -> FxHashMap<PropertyId, PropertyVerdict> {
        use crate::ay_bmc::{check_bmc, BmcConfig, BmcResult};

        let bmc_config = BmcConfig {
            max_depth: self.default_depth,
            solve_timeout: Some(time_budget),
            ..BmcConfig::default()
        };

        let mut verdicts = FxHashMap::default();
        match check_bmc(self.module, self.config, self.ctx, bmc_config) {
            Ok(BmcResult::Violation { ref trace, .. }) => {
                // BMC found a counterexample (invariant violation). SOUNDNESS
                // (fail closed): a violation is REPORTED only after the
                // explicit-state evaluator re-confirmed the counterexample; a
                // spurious SMT model leaves the properties Unknown for the
                // later (ultimately explicit BFS) phases.
                if crate::check::cross_validation::confirm_symbolic_cex_fail_closed(
                    self.module,
                    self.config,
                    trace,
                    crate::check::cross_validation::CrossValidationSource::Bmc,
                )
                .engine_agrees
                {
                    for prop in unresolved {
                        verdicts.insert(prop.clone(), PropertyVerdict::Violated);
                    }
                } else {
                    telemetry_eprintln!(
                        "[pipeline] BMC violation failed explicit-evaluator cross-validation \
                         — failing closed (properties stay Unknown)"
                    );
                }
            }
            Ok(BmcResult::Deadlock { .. }) => {
                // BMC found a reachable deadlock state — a property failure (a
                // reachable deadlock mirrors explicit-BFS Deadlock => Unsafe).
                // Deadlock witnesses come from sound concrete-state probing
                // (probe_deadlock_at_depth), not from a raw SMT model.
                for prop in unresolved {
                    verdicts.insert(prop.clone(), PropertyVerdict::Violated);
                }
            }
            Ok(BmcResult::BoundReached { .. }) | Ok(BmcResult::Unknown { .. }) => {
                // No bug within the bound or solver inconclusive. BMC cannot
                // prove safety, so leave as Unknown.
            }
            Err(e) => {
                eprintln!("Pipeline BMC error: {e}");
            }
        }
        verdicts
    }

    fn phase(&self) -> VerificationPhase {
        VerificationPhase::Bmc
    }
}

// ---------------------------------------------------------------------------
// PDR Runner (ay feature gate)
// ---------------------------------------------------------------------------

/// PDR runner wrapping `check_pdr_with_config()` for symbolic safety proving.
///
/// Maps:
/// - `PdrResult::Safe` -> `PropertyVerdict::Satisfied` for all unresolved properties
/// - `PdrResult::Unsafe` -> `PropertyVerdict::Violated` for all unresolved properties
/// - `PdrResult::Unknown` -> leaves as `Unknown`
///
/// Part of #3723.
#[cfg(feature = "ay")]
pub struct PdrRunner<'a> {
    module: &'a tla_core::ast::Module,
    config: &'a crate::config::Config,
    ctx: &'a crate::eval::EvalCtx,
}

#[cfg(feature = "ay")]
impl<'a> PdrRunner<'a> {
    /// Construct a PDR runner.
    pub fn new(
        module: &'a tla_core::ast::Module,
        config: &'a crate::config::Config,
        ctx: &'a crate::eval::EvalCtx,
    ) -> Self {
        Self {
            module,
            config,
            ctx,
        }
    }
}

#[cfg(feature = "ay")]
impl std::fmt::Debug for PdrRunner<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PdrRunner")
            .field("module", &self.module.name.node)
            .finish()
    }
}

#[cfg(feature = "ay")]
impl PhaseRunner for PdrRunner<'_> {
    fn run(
        &mut self,
        unresolved: &[PropertyId],
        time_budget: Duration,
    ) -> FxHashMap<PropertyId, PropertyVerdict> {
        use crate::ay_pdr::{check_pdr_with_config, PdrResult};
        use tla_ay::PdrConfig;

        let mut pdr_config = PdrConfig::default();
        pdr_config.solve_timeout = Some(time_budget);

        let mut verdicts = FxHashMap::default();
        match check_pdr_with_config(self.module, self.config, self.ctx, pdr_config) {
            Ok(PdrResult::Safe { .. }) => {
                // PDR proved all invariants hold for all reachable states.
                for prop in unresolved {
                    verdicts.insert(prop.clone(), PropertyVerdict::Satisfied);
                }
            }
            Ok(PdrResult::Unsafe { ref trace }) => {
                // PDR found a counterexample. SOUNDNESS (fail closed): report a
                // violation only after the explicit-state evaluator re-confirmed
                // it; a spurious CHC model leaves the properties Unknown for
                // the later (ultimately explicit BFS) phases.
                let bmc_states = crate::check::cross_validation::pdr_trace_to_bmc_states(trace);
                if crate::check::cross_validation::confirm_symbolic_cex_fail_closed(
                    self.module,
                    self.config,
                    &bmc_states,
                    crate::check::cross_validation::CrossValidationSource::Pdr,
                )
                .engine_agrees
                {
                    for prop in unresolved {
                        verdicts.insert(prop.clone(), PropertyVerdict::Violated);
                    }
                } else {
                    telemetry_eprintln!(
                        "[pipeline] PDR unsafe trace failed explicit-evaluator \
                         cross-validation — failing closed (properties stay Unknown)"
                    );
                }
            }
            Ok(PdrResult::Unknown { .. }) => {
                // Inconclusive.
            }
            Err(e) => {
                eprintln!("Pipeline PDR error: {e}");
            }
        }
        verdicts
    }

    fn phase(&self) -> VerificationPhase {
        VerificationPhase::Pdr
    }
}

// ---------------------------------------------------------------------------
// K-Induction Runner (ay feature gate)
// ---------------------------------------------------------------------------

/// K-induction runner wrapping `check_kinduction()` for symbolic safety proving.
///
/// Maps:
/// - `KInductionResult::Proved` -> `PropertyVerdict::Satisfied` for all unresolved properties
/// - `KInductionResult::Counterexample` -> `PropertyVerdict::Violated` for all unresolved properties
/// - `KInductionResult::Unknown` -> leaves as `Unknown`
///
/// Part of #3722.
#[cfg(feature = "ay")]
pub struct KInductionRunner<'a> {
    module: &'a tla_core::ast::Module,
    config: &'a crate::config::Config,
    ctx: &'a crate::eval::EvalCtx,
    /// Maximum induction depth (pipeline default: 20).
    max_k: usize,
}

#[cfg(feature = "ay")]
impl<'a> KInductionRunner<'a> {
    /// Construct a k-induction runner.
    ///
    /// `max_k` controls the maximum induction depth. A reasonable default
    /// is 10-20 for pipeline use.
    pub fn new(
        module: &'a tla_core::ast::Module,
        config: &'a crate::config::Config,
        ctx: &'a crate::eval::EvalCtx,
        max_k: usize,
    ) -> Self {
        Self {
            module,
            config,
            ctx,
            max_k,
        }
    }
}

#[cfg(feature = "ay")]
impl std::fmt::Debug for KInductionRunner<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KInductionRunner")
            .field("module", &self.module.name.node)
            .field("max_k", &self.max_k)
            .finish()
    }
}

#[cfg(feature = "ay")]
impl PhaseRunner for KInductionRunner<'_> {
    fn run(
        &mut self,
        unresolved: &[PropertyId],
        time_budget: Duration,
    ) -> FxHashMap<PropertyId, PropertyVerdict> {
        use crate::ay_kinduction::{check_kinduction, KInductionConfig, KInductionResult};

        let kind_config = KInductionConfig {
            max_k: self.max_k,
            solve_timeout: Some(time_budget),
            ..KInductionConfig::default()
        };

        let mut verdicts = FxHashMap::default();
        match check_kinduction(self.module, self.config, self.ctx, kind_config) {
            Ok(KInductionResult::Proved { .. }) => {
                // K-induction proved all invariants hold for all reachable states.
                for prop in unresolved {
                    verdicts.insert(prop.clone(), PropertyVerdict::Satisfied);
                }
            }
            Ok(KInductionResult::Counterexample { ref trace, .. }) => {
                // BMC base case found a counterexample. SOUNDNESS (fail
                // closed): report a violation only after the explicit-state
                // evaluator re-confirmed it; a spurious base-case model leaves
                // the properties Unknown for the later phases — a lane whose
                // base case it could not discharge never contributes a verdict.
                if crate::check::cross_validation::confirm_symbolic_cex_fail_closed(
                    self.module,
                    self.config,
                    trace,
                    crate::check::cross_validation::CrossValidationSource::KInduction,
                )
                .engine_agrees
                {
                    for prop in unresolved {
                        verdicts.insert(prop.clone(), PropertyVerdict::Violated);
                    }
                } else {
                    telemetry_eprintln!(
                        "[pipeline] k-Induction base-case counterexample failed \
                         explicit-evaluator cross-validation — failing closed \
                         (properties stay Unknown)"
                    );
                }
            }
            Ok(KInductionResult::Unknown { .. }) => {
                // Inconclusive — leave as Unknown for subsequent phases.
            }
            Err(e) => {
                eprintln!("Pipeline k-induction error: {e}");
            }
        }
        verdicts
    }

    fn phase(&self) -> VerificationPhase {
        VerificationPhase::KInduction
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::test_support::parse_module;
    use rustc_hash::FxHashMap;
    use std::time::Duration;

    use super::super::pipeline::{PhaseConfig, VerificationPipeline};

    // -------------------------------------------------------------------
    // Phase accessor smoke tests
    // -------------------------------------------------------------------

    #[test]
    fn test_random_walk_runner_phase() {
        // Verify the phase() accessor returns the correct variant.
        // Full integration tests require a TLA+ module, which is covered
        // by random_walk.rs tests. This test validates the PhaseRunner
        // contract at the type level.
        assert_eq!(VerificationPhase::RandomWalk.to_string(), "RandomWalk");
    }

    #[test]
    fn test_bfs_runner_phase() {
        assert_eq!(VerificationPhase::Bfs.to_string(), "BFS");
    }

    #[cfg(feature = "ay")]
    #[test]
    fn test_bmc_runner_phase() {
        assert_eq!(VerificationPhase::Bmc.to_string(), "BMC");
    }

    #[cfg(feature = "ay")]
    #[test]
    fn test_pdr_runner_phase() {
        assert_eq!(VerificationPhase::Pdr.to_string(), "PDR");
    }

    #[cfg(feature = "ay")]
    #[test]
    fn test_kinduction_runner_phase() {
        assert_eq!(VerificationPhase::KInduction.to_string(), "k-induction");
    }

    // -------------------------------------------------------------------
    // Helper: build Config with Init/Next/Invariants
    // -------------------------------------------------------------------

    fn config_with_invariants(init: &str, next: &str, invariants: &[&str]) -> Config {
        Config {
            init: Some(init.to_string()),
            next: Some(next.to_string()),
            invariants: invariants.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    // -------------------------------------------------------------------
    // End-to-end: BFS runner resolves satisfied invariant
    // -------------------------------------------------------------------

    /// Verify that BfsRunner correctly wraps the model checker and reports
    /// `Satisfied` when all reachable states satisfy the invariant.
    #[test]
    fn test_bfs_runner_e2e_invariant_satisfied() {
        let src = r#"
---- MODULE BfsE2eSafe ----
VARIABLE x
Init == x \in {0, 1, 2}
Next == x' = x
TypeOK == x \in {0, 1, 2}
====
"#;
        let module = parse_module(src);
        let config = config_with_invariants("Init", "Next", &["TypeOK"]);

        let mut runner = BfsRunner::new(&module, &[], &config);
        let properties = vec!["TypeOK".to_string()];
        let verdicts = runner.run(&properties, Duration::from_secs(30));

        assert_eq!(
            verdicts.get("TypeOK"),
            Some(&PropertyVerdict::Satisfied),
            "BFS should report invariant as Satisfied for trivially safe spec"
        );
    }

    // -------------------------------------------------------------------
    // End-to-end: BFS runner detects violated invariant
    // -------------------------------------------------------------------

    /// Verify that BfsRunner correctly detects an invariant violation and
    /// reports `Violated` for the offending property.
    #[test]
    fn test_bfs_runner_e2e_invariant_violated() {
        let src = r#"
---- MODULE BfsE2eUnsafe ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
SmallBound == x < 3
====
"#;
        let module = parse_module(src);
        let config = config_with_invariants("Init", "Next", &["SmallBound"]);

        let mut runner = BfsRunner::new(&module, &[], &config);
        let properties = vec!["SmallBound".to_string()];
        let verdicts = runner.run(&properties, Duration::from_secs(30));

        assert_eq!(
            verdicts.get("SmallBound"),
            Some(&PropertyVerdict::Violated),
            "BFS should detect invariant violation when x reaches 3"
        );
    }

    // -------------------------------------------------------------------
    // End-to-end: BFS runner surfaces a reached deadlock out-of-band
    // -------------------------------------------------------------------

    /// A reachable deadlock leaves the per-invariant verdicts Unknown (it
    /// invalidates no single named invariant), but the runner must record it via
    /// `deadlock_reached()` so the pipeline/CLI reports it instead of silently
    /// exiting 0 (the `--strategy`/`--pipeline` missed-deadlock bug).
    #[test]
    fn test_bfs_runner_e2e_deadlock_signalled() {
        let src = r#"
---- MODULE BfsE2eDeadlock ----
VARIABLE x
Init == x = 0
Next == x = 0 /\ x' = 1
Holds == x >= 0
====
"#;
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Holds".to_string()],
            check_deadlock: true,
            ..Default::default()
        };

        let mut runner = BfsRunner::new(&module, &[], &config);
        let properties = vec!["Holds".to_string()];
        let verdicts = runner.run(&properties, Duration::from_secs(30));

        assert!(
            runner.deadlock_reached(),
            "BFS must record the reachable deadlock (x=1 is terminal)"
        );
        // The holding invariant is NOT reported Satisfied — a deadlock-reaching
        // run proves nothing about the per-invariant verdict; it is left Unknown.
        assert_ne!(
            verdicts.get("Holds"),
            Some(&PropertyVerdict::Satisfied),
            "a deadlock-reaching run must not claim the invariant Satisfied"
        );
    }

    // -------------------------------------------------------------------
    // End-to-end: RandomWalk runner detects violation
    // -------------------------------------------------------------------

    /// Verify that RandomWalkRunner can find an easy invariant violation
    /// via random exploration.
    #[test]
    fn test_random_walk_runner_e2e_violation() {
        let src = r#"
---- MODULE WalkE2eUnsafe ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
SmallBound == x < 3
====
"#;
        let module = parse_module(src);
        let config = config_with_invariants("Init", "Next", &["SmallBound"]);
        let runtime_config = config.runtime_model_config();

        let checker = ModelChecker::new(&module, &runtime_config);
        let walk_config = RandomWalkConfig {
            num_walks: 100,
            max_depth: 100,
            seed: Some(42),
        };
        let mut runner = RandomWalkRunner::new(checker, walk_config);

        let properties = vec!["SmallBound".to_string()];
        let verdicts = runner.run(&properties, Duration::from_secs(5));

        assert_eq!(
            verdicts.get("SmallBound"),
            Some(&PropertyVerdict::Violated),
            "Random walk should easily find violation when x reaches 3 in 100 steps"
        );
    }

    // -------------------------------------------------------------------
    // End-to-end: full pipeline resolves properties across phases
    // -------------------------------------------------------------------

    /// End-to-end pipeline test: RandomWalk(5s) + BFS(300s) together resolve
    /// all properties. This validates the pipeline orchestration with real
    /// verification backends, not mocks.
    ///
    /// Spec has two invariants:
    /// - `EasyViolation`: violated quickly (x reaches 3 in 3 steps)
    /// - Both should be resolved by BFS if not by random walk.
    #[test]
    fn test_pipeline_e2e_resolves_all_properties() {
        let src = r#"
---- MODULE PipelineE2e ----
VARIABLE x
Init == x \in {0, 1}
Next == IF x < 5 THEN x' = x + 1 ELSE x' = x
Bound == x <= 10
====
"#;
        let module = parse_module(src);
        let config = config_with_invariants("Init", "Next", &["Bound"]);
        let runtime_config = config.runtime_model_config();

        let properties = vec!["Bound".to_string()];

        // Build pipeline: RandomWalk -> BFS (skip symbolic phases for simplicity).
        let pipeline = VerificationPipeline::new(vec![
            PhaseConfig {
                phase: VerificationPhase::RandomWalk,
                time_budget: Duration::from_secs(5),
                enabled: true,
            },
            PhaseConfig {
                phase: VerificationPhase::Bfs,
                time_budget: Duration::from_secs(60),
                enabled: true,
            },
        ]);

        let mut runners: FxHashMap<VerificationPhase, Box<dyn PhaseRunner>> = FxHashMap::default();

        // RandomWalk runner
        let walk_checker = ModelChecker::new(&module, &runtime_config);
        let walk_config = RandomWalkConfig {
            num_walks: 50,
            max_depth: 100,
            seed: Some(42),
        };
        runners.insert(
            VerificationPhase::RandomWalk,
            Box::new(RandomWalkRunner::new(walk_checker, walk_config)),
        );

        // BFS runner
        runners.insert(
            VerificationPhase::Bfs,
            Box::new(BfsRunner::new(&module, &[], &config)),
        );

        let result = pipeline.run(&properties, &mut runners);

        // The invariant `Bound` (x <= 10) holds for this spec since x only
        // goes up to 5. BFS should verify it as Satisfied.
        assert_eq!(
            result.verdicts.get("Bound"),
            Some(&PropertyVerdict::Satisfied),
            "Pipeline should resolve Bound as Satisfied (x never exceeds 5)"
        );
        // At least one phase should have run.
        assert!(
            !result.phases_run.is_empty(),
            "Pipeline should have run at least one phase"
        );
    }

    /// End-to-end pipeline test with a violation: the pipeline should detect
    /// the violation in an early phase and skip expensive later phases.
    #[test]
    fn test_pipeline_e2e_early_violation_detection() {
        let src = r#"
---- MODULE PipelineViolation ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
TooSmall == x < 3
====
"#;
        let module = parse_module(src);
        let config = config_with_invariants("Init", "Next", &["TooSmall"]);
        let runtime_config = config.runtime_model_config();

        let properties = vec!["TooSmall".to_string()];

        // Two-phase pipeline: RandomWalk -> BFS.
        let pipeline = VerificationPipeline::new(vec![
            PhaseConfig {
                phase: VerificationPhase::RandomWalk,
                time_budget: Duration::from_secs(5),
                enabled: true,
            },
            PhaseConfig {
                phase: VerificationPhase::Bfs,
                time_budget: Duration::from_secs(60),
                enabled: true,
            },
        ]);

        let mut runners: FxHashMap<VerificationPhase, Box<dyn PhaseRunner>> = FxHashMap::default();

        // RandomWalk runner — should find the violation in 3 steps.
        let walk_checker = ModelChecker::new(&module, &runtime_config);
        let walk_config = RandomWalkConfig {
            num_walks: 100,
            max_depth: 100,
            seed: Some(42),
        };
        runners.insert(
            VerificationPhase::RandomWalk,
            Box::new(RandomWalkRunner::new(walk_checker, walk_config)),
        );

        // BFS runner (should be skipped if RandomWalk resolves everything).
        runners.insert(
            VerificationPhase::Bfs,
            Box::new(BfsRunner::new(&module, &[], &config)),
        );

        let result = pipeline.run(&properties, &mut runners);

        // The violation should be detected.
        assert_eq!(
            result.verdicts.get("TooSmall"),
            Some(&PropertyVerdict::Violated),
            "Pipeline should detect that x reaches 3, violating TooSmall"
        );

        // RandomWalk should resolve it, so only 1 phase should run (early exit).
        assert_eq!(
            result.phases_run.len(),
            1,
            "Pipeline should early-exit after RandomWalk finds the violation"
        );
        assert_eq!(
            result.phases_run[0].phase,
            VerificationPhase::RandomWalk,
            "Only RandomWalk should have run"
        );
        assert_eq!(
            result.phases_run[0].properties_resolved, 1,
            "RandomWalk should resolve exactly 1 property"
        );
    }

    // -------------------------------------------------------------------
    // Regression #3: BFS wall-clock deadline backstop
    // -------------------------------------------------------------------

    /// Regression for bug #3 (HIGH overrun/OOM): the BFS pipeline runner used
    /// to drop its `_time_budget`, so `ty check` had no wall-clock backstop —
    /// an unbounded/explosive spec would run until manual interruption or OOM.
    ///
    /// This spec has a >10k-state space (two vars over 0..100). With a deadline
    /// already in the past, the unified BFS worker loop must hit the deadline
    /// poll and stop CLEANLY with a partial result — leaving every property
    /// Unknown (fail-closed), NEVER reporting `Satisfied` on a partial space and
    /// NEVER running to completion.
    ///
    /// The deadline is enforced by validation/poll, not by actually spending
    /// wall-clock time: we set an already-elapsed deadline, so the very first
    /// `DEADLINE_CHECK_INTERVAL` poll fires and the run returns immediately.
    #[test]
    #[cfg_attr(test, ntest::timeout(30000))]
    fn test_bfs_runner_deadline_returns_clean_partial_not_success() {
        // 101 * 101 = 10201 reachable states — well over the 4096-state deadline
        // poll interval, so the poll is guaranteed to fire before exhaustion.
        let src = r#"
---- MODULE BfsDeadlinePartial ----
VARIABLE x, y
Init == x = 0 /\ y = 0
Next == \/ (x' = (x + 1) % 101 /\ y' = y)
        \/ (y' = (y + 1) % 101 /\ x' = x)
AlwaysTrue == x >= 0 /\ y >= 0
====
"#;
        let module = parse_module(src);
        let config = config_with_invariants("Init", "Next", &["AlwaysTrue"]);

        let mut runner = BfsRunner::new(&module, &[], &config);
        let properties = vec!["AlwaysTrue".to_string()];

        // A 1ns budget => the deadline is essentially in the past by the time
        // the first poll runs. The run must DECLINE (leave Unknown), not prove
        // the invariant from a partial exploration.
        let verdicts = runner.run(&properties, Duration::from_nanos(1));

        assert_eq!(
            verdicts.get("AlwaysTrue"),
            None,
            "Deadline-interrupted BFS must leave the property Unknown (fail-closed), \
             never report Satisfied from a partial state space"
        );
    }

    /// Companion direct-checker regression: assert the actual `CheckResult` is
    /// `LimitReached {{ Time }}` (a sound partial), not `Success`. This pins the
    /// soundness contract at the `ModelChecker::check()` boundary so a future
    /// refactor that re-routes the verdict cannot silently turn a timed-out run
    /// into a false proof of correctness.
    #[test]
    #[cfg_attr(test, ntest::timeout(30000))]
    fn test_model_checker_deadline_yields_limit_reached_time() {
        use crate::check::{CheckResult, LimitType};

        let src = r#"
---- MODULE BfsDeadlineLimit ----
VARIABLE x, y
Init == x = 0 /\ y = 0
Next == \/ (x' = (x + 1) % 101 /\ y' = y)
        \/ (y' = (y + 1) % 101 /\ x' = x)
AlwaysTrue == x >= 0 /\ y >= 0
====
"#;
        let module = parse_module(src);
        let config = config_with_invariants("Init", "Next", &["AlwaysTrue"]);
        let runtime_config = config.runtime_model_config();

        let mut checker = ModelChecker::new(&module, &runtime_config);
        // Already-elapsed deadline => first poll (state 4096) stops the run.
        checker.set_time_budget(Duration::from_nanos(1));
        let result = checker.check();

        match result {
            CheckResult::LimitReached {
                limit_type: LimitType::Time,
                ..
            } => {}
            CheckResult::Success(_) => panic!(
                "UNSOUND: deadline-interrupted BFS reported Success on a partial state space"
            ),
            other => panic!("expected LimitReached {{ Time }} on deadline, got {other:?}"),
        }
    }
}
