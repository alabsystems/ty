// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Portfolio configuration types: engine variants, portfolio config, and results.

use std::time::Duration;

use crate::check_result::CheckResult;
use crate::ic3::Ic3Config;

/// Configuration for a single engine in the portfolio.
#[derive(Debug, Clone)]
pub enum EngineConfig {
    /// BMC with a given step size (ay-sat default config).
    Bmc {
        /// Number of depths unrolled between consecutive SAT checks.
        step: usize,
    },
    /// BMC with a size-adaptive step (ty ladder policy).
    ///
    /// The step is chosen at runtime as the largest rung of the BMC engine's
    /// step ladder {1, 10, 100, 500} whose unroll window (estimated clauses
    /// loaded between SAT checks) fits the engine's clause budget — see
    /// `BmcEngine::new_dynamic`. Small circuits get large steps; large
    /// circuits get step=1.
    BmcDynamic,
    /// BMC with a ay-sat configuration variant and given step size.
    ///
    /// Portfolio diversity comes from ay-sat's configuration knobs: different
    /// restart policies (Luby, geometric, stable-only), branching heuristics
    /// (VMTF, CHB), and preprocessing toggles. This replaces the former
    /// CaDiCaL BMC backend — we own ay-sat and do not use external solvers.
    BmcAYVariant {
        /// Number of depths unrolled between consecutive SAT checks.
        step: usize,
        /// ay-sat configuration variant (restart/branching/preprocessing knobs).
        backend: crate::sat_types::SolverBackend,
    },
    /// BMC with a ay-sat variant and the size-adaptive step (ty ladder policy).
    BmcAYVariantDynamic {
        /// ay-sat configuration variant.
        backend: crate::sat_types::SolverBackend,
    },
    /// BMC with geometric backoff step sizing (#4123).
    ///
    /// Starts at step=1 for the first `initial_depths` depths (thorough shallow
    /// coverage), then doubles the step size every `double_interval` SAT calls,
    /// capped at `max_step`. This reaches deep counterexamples much faster than
    /// fixed step=1 while still catching shallow bugs.
    ///
    /// Designed for Sokoban/microban puzzles whose counterexamples sit at
    /// depth 100+ and reward large-step unrolling.
    BmcGeometricBackoff {
        /// Number of initial depths checked one-by-one (step size 1).
        initial_depths: usize,
        /// Number of SAT calls between each doubling of the step size.
        double_interval: usize,
        /// Upper bound on the step size after repeated doubling.
        max_step: usize,
    },
    /// BMC with geometric backoff and a ay-sat configuration variant.
    BmcGeometricBackoffAYVariant {
        /// Number of initial depths checked one-by-one (step size 1).
        initial_depths: usize,
        /// Number of SAT calls between each doubling of the step size.
        double_interval: usize,
        /// Upper bound on the step size after repeated doubling.
        max_step: usize,
        /// ay-sat configuration variant.
        backend: crate::sat_types::SolverBackend,
    },
    /// BMC with a linear-offset start depth for mid-deep SAT search (#4299, Wave 29).
    ///
    /// Skips the first `start_depth` depths (already covered by shallow step-1 BMC
    /// configs in the portfolio) and then runs linear step-1 BMC from `start_depth`
    /// to `max_depth`. Unlike geometric backoff, which overshoots by doubling,
    /// linear-offset checks every depth past the skip region — essential for
    /// Sokoban/microban SAT puzzles whose counterexample sits at a specific depth
    /// that geometric doubling skips over.
    ///
    /// Designed for HWMCC top-50 Tier 2 Sokoban SAT benchmarks (microban_64/77/89/
    /// 118/132/136/148/149) whose counterexamples sit at depth ~100-500.
    BmcLinearOffset {
        /// First depth at which to run the initial SAT check. Prior depths are
        /// unrolled (clauses loaded) but not checked, skipping redundant work
        /// already performed by step-1 BMC configs.
        start_depth: usize,
        /// Step between SAT checks after `start_depth`. Use 1 for exhaustive
        /// per-depth search.
        step: usize,
        /// Maximum depth to explore.
        max_depth: usize,
    },
    /// k-Induction.
    Kind,
    /// k-Induction with simple-path constraint.
    ///
    /// Asserts pairwise state distinctness in the induction trace,
    /// strengthening the hypothesis to prove harder properties.
    KindSimplePath,
    /// k-Induction with skip-bmc mode (induction step only).
    ///
    /// Only checks the inductive step, skipping the base case BMC check.
    /// Useful in portfolios where BMC is already running in a separate thread.
    /// This saves solver time by focusing purely on proving the property
    /// k-inductive.
    KindSkipBmc,
    /// k-Induction with a ay-sat configuration variant.
    ///
    /// Portfolio diversity: different ay-sat configs race against each other
    /// on the same k-induction problem.
    KindAYVariant {
        /// ay-sat configuration variant.
        backend: crate::sat_types::SolverBackend,
    },
    /// k-Induction with a ay-sat variant and skip-bmc mode.
    KindSkipBmcAYVariant {
        /// ay-sat configuration variant.
        backend: crate::sat_types::SolverBackend,
    },
    /// IC3/PDR with default configuration (all optimizations off).
    Ic3,
    /// IC3/PDR with a specific configuration and human-readable name.
    Ic3Configured {
        /// IC3 tuning configuration.
        config: Ic3Config,
        /// Label used in portfolio diagnostics and the winning-engine name.
        name: String,
    },
    /// CEGAR-IC3: IC3 inside a counterexample-guided abstraction refinement loop.
    ///
    /// Starts with an abstract model (COI latches only), runs IC3, and refines
    /// if the counterexample is spurious. Effective on large circuits where most
    /// latches are irrelevant to the property.
    ///
    /// The `mode` controls how aggressively the abstraction removes information:
    /// - `AbstractConstraints` (cegar-const): only relax constraint enforcement
    /// - `AbstractAll` (cegar-full): remove both constraints and transition relation
    CegarIc3 {
        /// IC3 tuning configuration used on each refinement round.
        config: Ic3Config,
        /// Label used in portfolio diagnostics and the winning-engine name.
        name: String,
        /// How aggressively the abstraction discards constraints/transitions.
        mode: crate::ic3::cegar::AbstractionMode,
    },
    /// Strengthened k-Induction with auxiliary invariant discovery.
    KindStrengthened,
    /// Strengthened k-Induction with a ay-sat configuration variant.
    KindStrengthenedAYVariant {
        /// ay-sat configuration variant.
        backend: crate::sat_types::SolverBackend,
    },
    /// Random forward simulation: SAT-free exploration with random inputs.
    ///
    /// Simulates the circuit forward from the initial state with random inputs,
    /// checking if a bad state is reached. Extremely fast (millions of steps/sec)
    /// but probabilistic — will not find bugs that require specific input sequences.
    ///
    /// Most effective on circuits where the bad state is reachable via many
    /// different input paths. For Sokoban puzzles (single specific solution),
    /// this engine will not help, but it provides zero-cost diversity in the
    /// portfolio since it requires no SAT calls.
    ///
    /// `steps_per_walk`: how many forward steps each random walk takes.
    /// `num_walks`: how many independent random walks to run.
    /// `seed`: random seed for reproducibility with portfolio diversity.
    RandomSim {
        /// Number of forward steps taken in each random walk.
        steps_per_walk: usize,
        /// Number of independent random walks to run.
        num_walks: usize,
        /// Seed for the random input generator (for reproducible diversity).
        seed: u64,
    },
    /// GPU exhaustive bounded model checking (`bmc/gpu_exhaustive.rs`).
    ///
    /// Unrolls the transition relation `k` steps into ONE combinational AIG and
    /// enumerates ALL free-variable assignments on the GPU. Unlike random
    /// simulation (which only falsifies), a "no bad" result is a COMPLETE
    /// bounded-safety PROOF (`BoundedSafe`) — but bounded, so it is surfaced as
    /// `Unknown` (never a full `Safe`; that needs k-induction). An `Unsafe`
    /// result is re-derived through the CPU BMC engine to obtain a
    /// portfolio-verifiable counterexample trace.
    ///
    /// The carrier declines (→ CPU BMC fallback) on a non-CUDA host, relational
    /// init, any invariant constraint, or when `(k+1)·num_inputs` exceeds the
    /// exhaustive free-variable cap. Purely additive and fail-closed.
    GpuExhaustiveBmc {
        /// Maximum unroll depth `k` enumerated on the GPU (capped by
        /// `max_depth` at run time).
        max_k: usize,
    },
    /// BDD symbolic reachability (`bdd_reach.rs`): the workspace's general
    /// ROBDD engine (GC + sifting + cooperative abort) computes the EXACT
    /// forward reachable latch set.
    ///
    /// Unlike every SAT lane, a converged fixpoint decides UNBOUNDED safety
    /// outright: `Safe` is an exact proof (re-verified inductively inside the
    /// engine before surfacing), and a bad intersection reports the MINIMAL
    /// counterexample depth, which the runner re-derives through the CPU BMC
    /// engine for a portfolio-verifiable trace (the GPU-lane protocol). The
    /// lane declines fail-closed on constraints, relational init, size caps,
    /// or budget exhaustion — purely additive to the portfolio.
    BddReach {
        /// Admission caps and node budget.
        config: crate::bdd_reach::BddReachConfig,
    },
}

impl EngineConfig {
    /// Human-readable name for this engine configuration.
    pub fn name(&self) -> &str {
        match self {
            EngineConfig::Bmc { step } => {
                // Return a static str for common cases; callers can format themselves
                if *step == 1 {
                    "bmc-1"
                } else {
                    "bmc"
                }
            }
            EngineConfig::BmcDynamic => "bmc-dynamic",
            EngineConfig::BmcAYVariant { backend, .. } => match backend {
                crate::sat_types::SolverBackend::AYLuby => "bmc-ay-luby",
                crate::sat_types::SolverBackend::AYStable => "bmc-ay-stable",
                crate::sat_types::SolverBackend::AYGeometric => "bmc-ay-geometric",
                crate::sat_types::SolverBackend::AYVmtf => "bmc-ay-vmtf",
                crate::sat_types::SolverBackend::AYChb => "bmc-ay-chb",
                crate::sat_types::SolverBackend::AYNoPreprocess => "bmc-ay-nopreproc",
                crate::sat_types::SolverBackend::Simple => "bmc-simple",
                _ => "bmc-ay-variant",
            },
            EngineConfig::BmcAYVariantDynamic { backend } => match backend {
                crate::sat_types::SolverBackend::AYLuby => "bmc-ay-luby-dynamic",
                crate::sat_types::SolverBackend::AYStable => "bmc-ay-stable-dynamic",
                _ => "bmc-ay-variant-dynamic",
            },
            EngineConfig::BmcGeometricBackoff { .. } => "bmc-geometric-backoff",
            EngineConfig::BmcGeometricBackoffAYVariant { backend, .. } => match backend {
                crate::sat_types::SolverBackend::AYLuby => "bmc-geometric-ay-luby",
                crate::sat_types::SolverBackend::AYStable => "bmc-geometric-ay-stable",
                crate::sat_types::SolverBackend::Simple => "bmc-geometric-simple",
                _ => "bmc-geometric-ay-variant",
            },
            EngineConfig::BmcLinearOffset { .. } => "bmc-linear-offset",
            EngineConfig::Kind => "kind",
            EngineConfig::KindSimplePath => "kind-simple-path",
            EngineConfig::KindSkipBmc => "kind-skip-bmc",
            EngineConfig::KindAYVariant { backend } => match backend {
                crate::sat_types::SolverBackend::AYLuby => "kind-ay-luby",
                crate::sat_types::SolverBackend::AYStable => "kind-ay-stable",
                crate::sat_types::SolverBackend::AYVmtf => "kind-ay-vmtf",
                _ => "kind-ay-variant",
            },
            EngineConfig::KindSkipBmcAYVariant { backend } => match backend {
                crate::sat_types::SolverBackend::AYLuby => "kind-skip-bmc-ay-luby",
                crate::sat_types::SolverBackend::AYStable => "kind-skip-bmc-ay-stable",
                crate::sat_types::SolverBackend::AYVmtf => "kind-skip-bmc-ay-vmtf",
                _ => "kind-skip-bmc-ay-variant",
            },
            EngineConfig::Ic3 => "ic3-default",
            EngineConfig::Ic3Configured { name, .. } => name.as_str(),
            EngineConfig::CegarIc3 { name, .. } => name.as_str(),
            EngineConfig::KindStrengthened => "kind-strengthened",
            EngineConfig::KindStrengthenedAYVariant { backend } => match backend {
                crate::sat_types::SolverBackend::AYLuby => "kind-str-ay-luby",
                crate::sat_types::SolverBackend::AYStable => "kind-str-ay-stable",
                _ => "kind-str-ay-variant",
            },
            EngineConfig::RandomSim { .. } => "random-sim",
            EngineConfig::GpuExhaustiveBmc { .. } => "gpu-exhaustive-bmc",
            EngineConfig::BddReach { .. } => "bdd-reach",
        }
    }

    /// Stable coarse engine class for capability-report inventory rows.
    pub fn kind_code(&self) -> &'static str {
        match self {
            EngineConfig::Bmc { .. }
            | EngineConfig::BmcDynamic
            | EngineConfig::BmcAYVariant { .. }
            | EngineConfig::BmcAYVariantDynamic { .. }
            | EngineConfig::BmcGeometricBackoff { .. }
            | EngineConfig::BmcGeometricBackoffAYVariant { .. }
            | EngineConfig::BmcLinearOffset { .. }
            | EngineConfig::GpuExhaustiveBmc { .. } => "bmc",
            EngineConfig::Kind
            | EngineConfig::KindSimplePath
            | EngineConfig::KindSkipBmc
            | EngineConfig::KindAYVariant { .. }
            | EngineConfig::KindSkipBmcAYVariant { .. }
            | EngineConfig::KindStrengthened
            | EngineConfig::KindStrengthenedAYVariant { .. } => "kinduction",
            EngineConfig::Ic3 | EngineConfig::Ic3Configured { .. } => "ic3",
            EngineConfig::CegarIc3 { .. } => "cegar_ic3",
            EngineConfig::RandomSim { .. } => "random_sim",
            EngineConfig::BddReach { .. } => "bdd_reach",
        }
    }

    /// Parameterized label for release/diagnose evidence.
    ///
    /// `name()` intentionally preserves historical solver attribution strings,
    /// so several variants share a generic name. This label is inventory-only:
    /// it distinguishes same-name configurations without changing execution.
    pub fn diagnostic_label(&self) -> String {
        match self {
            EngineConfig::Bmc { step } => format!("bmc-step-{step}"),
            EngineConfig::BmcDynamic => "bmc-dynamic".into(),
            EngineConfig::BmcAYVariant { step, backend } => {
                format!("bmc-step-{step}-{}", solver_backend_code(*backend))
            }
            EngineConfig::BmcAYVariantDynamic { backend } => {
                format!("bmc-dynamic-{}", solver_backend_code(*backend))
            }
            EngineConfig::BmcGeometricBackoff {
                initial_depths,
                double_interval,
                max_step,
            } => format!(
                "bmc-geometric-initial-{initial_depths}-interval-{double_interval}-max-{max_step}"
            ),
            EngineConfig::BmcGeometricBackoffAYVariant {
                initial_depths,
                double_interval,
                max_step,
                backend,
            } => format!(
                "bmc-geometric-initial-{initial_depths}-interval-{double_interval}-max-{max_step}-{}",
                solver_backend_code(*backend)
            ),
            EngineConfig::BmcLinearOffset {
                start_depth,
                step,
                max_depth,
            } => {
                format!("bmc-linear-offset-start-{start_depth}-step-{step}-max-{max_depth}")
            }
            EngineConfig::Kind => "kind".into(),
            EngineConfig::KindSimplePath => "kind-simple-path".into(),
            EngineConfig::KindSkipBmc => "kind-skip-bmc".into(),
            EngineConfig::KindAYVariant { backend } => {
                format!("kind-{}", solver_backend_code(*backend))
            }
            EngineConfig::KindSkipBmcAYVariant { backend } => {
                format!("kind-skip-bmc-{}", solver_backend_code(*backend))
            }
            EngineConfig::KindStrengthened => "kind-strengthened".into(),
            EngineConfig::KindStrengthenedAYVariant { backend } => {
                format!("kind-strengthened-{}", solver_backend_code(*backend))
            }
            EngineConfig::Ic3 => "ic3-default".into(),
            EngineConfig::Ic3Configured { name, config } => {
                format!("{name}-seed-{}", config.random_seed)
            }
            EngineConfig::CegarIc3 { name, mode, .. } => {
                format!("{name}-{}", cegar_mode_code(*mode))
            }
            EngineConfig::RandomSim {
                steps_per_walk,
                num_walks,
                seed,
            } => format!("random-sim-steps-{steps_per_walk}-walks-{num_walks}-seed-{seed}"),
            EngineConfig::GpuExhaustiveBmc { max_k } => format!("gpu-exhaustive-bmc-k-{max_k}"),
            EngineConfig::BddReach { config } => format!(
                "bdd-reach-latches-{}-ands-{}",
                config.max_latches, config.max_ands
            ),
        }
    }

    /// Disable IC3 consecutive-Unknown backend escalation for controlled sweeps.
    ///
    /// The plain `Ic3` variant carries default config implicitly, so disabling
    /// the fallback promotes it to an explicit configured IC3 engine. Other
    /// non-IC3 engines are left unchanged.
    pub fn with_ic3_unknown_fallback_disabled(self) -> Self {
        match self {
            EngineConfig::Ic3 => EngineConfig::Ic3Configured {
                config: Ic3Config {
                    unknown_fallback_enabled: false,
                    ..Ic3Config::default()
                },
                name: "ic3-default-no-unknown-fallback".into(),
            },
            EngineConfig::Ic3Configured { mut config, name } => {
                config.unknown_fallback_enabled = false;
                EngineConfig::Ic3Configured { config, name }
            }
            EngineConfig::CegarIc3 {
                mut config,
                name,
                mode,
            } => {
                config.unknown_fallback_enabled = false;
                EngineConfig::CegarIc3 { config, name, mode }
            }
            other => other,
        }
    }
}

fn solver_backend_code(backend: crate::sat_types::SolverBackend) -> &'static str {
    match backend {
        crate::sat_types::SolverBackend::AYSat => "ay-sat",
        crate::sat_types::SolverBackend::AYLuby => "ay-luby",
        crate::sat_types::SolverBackend::AYStable => "ay-stable",
        crate::sat_types::SolverBackend::AYGeometric => "ay-geometric",
        crate::sat_types::SolverBackend::AYVmtf => "ay-vmtf",
        crate::sat_types::SolverBackend::AYChb => "ay-chb",
        crate::sat_types::SolverBackend::AYNoPreprocess => "ay-no-preprocess",
        crate::sat_types::SolverBackend::Simple => "simple",
    }
}

fn cegar_mode_code(mode: crate::ic3::cegar::AbstractionMode) -> &'static str {
    match mode {
        crate::ic3::cegar::AbstractionMode::AbstractConstraints => "abstract-constraints",
        crate::ic3::cegar::AbstractionMode::AbstractAll => "abstract-all",
    }
}

/// Result of a portfolio run, including which solver produced the answer.
#[derive(Debug, Clone)]
pub struct PortfolioResult {
    /// The model checking result.
    pub result: CheckResult,
    /// Name of the solver configuration that produced this result.
    pub solver_name: String,
    /// Wall-clock time taken by the winning solver.
    pub time_secs: f64,
}

/// Configuration for the portfolio solver.
#[derive(Debug, Clone)]
pub struct PortfolioConfig {
    /// Overall time budget.
    pub timeout: Duration,
    /// Engine configurations to run in parallel.
    pub engines: Vec<EngineConfig>,
    /// Maximum unrolling depth for BMC/k-induction.
    pub max_depth: usize,
    /// Preprocessing configuration (#4124).
    pub preprocess: crate::preprocess::PreprocessConfig,
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        super::factory::default_portfolio()
    }
}

impl PortfolioConfig {
    /// Return this portfolio with IC3 consecutive-Unknown fallback disabled.
    ///
    /// This is intentionally an explicit opt-in helper for #4233/#4494 sweep
    /// evidence. Default production construction keeps the fallback enabled.
    pub fn with_ic3_unknown_fallback_disabled(mut self) -> Self {
        self.engines = self
            .engines
            .into_iter()
            .map(EngineConfig::with_ic3_unknown_fallback_disabled)
            .collect();
        self
    }
}

/// Simple single-engine configuration (no parallelism).
pub fn single_bmc(timeout: Duration, max_depth: usize) -> PortfolioConfig {
    PortfolioConfig {
        timeout,
        engines: vec![EngineConfig::Bmc { step: 1 }],
        max_depth,
        preprocess: crate::preprocess::PreprocessConfig::default(),
    }
}

/// Single BDD symbolic-reachability engine (no parallelism) — for targeted
/// runs and the lane's own end-to-end tests. The BMC re-derivation of an
/// Unsafe verdict runs inside the same lane, so this configuration still
/// produces portfolio-verifiable traces.
pub fn single_bdd_reach(timeout: Duration) -> PortfolioConfig {
    PortfolioConfig {
        timeout,
        engines: vec![EngineConfig::BddReach {
            config: crate::bdd_reach::BddReachConfig::default(),
        }],
        max_depth: 10000,
        preprocess: crate::preprocess::PreprocessConfig::default(),
    }
}

/// Single IC3 engine with a specific configuration.
pub fn single_ic3(timeout: Duration, config: Ic3Config, name: &str) -> PortfolioConfig {
    PortfolioConfig {
        timeout,
        engines: vec![EngineConfig::Ic3Configured {
            config,
            name: name.into(),
        }],
        max_depth: 10000,
        preprocess: crate::preprocess::PreprocessConfig::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ic3_unknown_fallback_sweep_transform_converts_plain_ic3() {
        let engine = EngineConfig::Ic3.with_ic3_unknown_fallback_disabled();

        match engine {
            EngineConfig::Ic3Configured { config, name } => {
                assert_eq!(name, "ic3-default-no-unknown-fallback");
                assert!(!config.unknown_fallback_enabled);
            }
            other => panic!("expected configured IC3, got {other:?}"),
        }
    }

    #[test]
    fn ic3_unknown_fallback_sweep_transform_updates_configured_ic3_and_cegar_only() {
        let portfolio = PortfolioConfig {
            timeout: Duration::from_secs(10),
            engines: vec![
                EngineConfig::Bmc { step: 1 },
                EngineConfig::Ic3Configured {
                    config: Ic3Config {
                        ctp: true,
                        unknown_fallback_enabled: true,
                        ..Ic3Config::default()
                    },
                    name: "ic3-ctp-test".into(),
                },
                EngineConfig::CegarIc3 {
                    config: Ic3Config {
                        inf_frame: true,
                        unknown_fallback_enabled: true,
                        ..Ic3Config::default()
                    },
                    name: "cegar-test".into(),
                    mode: crate::ic3::cegar::AbstractionMode::AbstractAll,
                },
            ],
            max_depth: 25,
            preprocess: crate::preprocess::PreprocessConfig::default(),
        }
        .with_ic3_unknown_fallback_disabled();

        assert!(matches!(
            portfolio.engines[0],
            EngineConfig::Bmc { step: 1 }
        ));

        match &portfolio.engines[1] {
            EngineConfig::Ic3Configured { config, name } => {
                assert_eq!(name, "ic3-ctp-test");
                assert!(config.ctp);
                assert!(!config.unknown_fallback_enabled);
            }
            other => panic!("expected configured IC3, got {other:?}"),
        }

        match &portfolio.engines[2] {
            EngineConfig::CegarIc3 { config, name, mode } => {
                assert_eq!(name, "cegar-test");
                assert!(config.inf_frame);
                assert!(!config.unknown_fallback_enabled);
                assert_eq!(*mode, crate::ic3::cegar::AbstractionMode::AbstractAll);
            }
            other => panic!("expected CEGAR IC3, got {other:?}"),
        }
    }
}
