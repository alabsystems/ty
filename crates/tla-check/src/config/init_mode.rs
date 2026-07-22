// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! InitMode enum and ay feature flags for initial state enumeration.

/// Mode for initial state enumeration.
///
/// Controls whether to use brute-force enumeration or ay-based SMT solving.
/// This is primarily for testing and allows deterministic selection without
/// relying on environment variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InitMode {
    /// Use brute-force enumeration only (never ay)
    BruteForce,
    /// Use ay if analysis recommends it, fall back to brute-force if ay fails
    /// This is the default behavior.
    #[default]
    Auto,
    /// Always try ay first, fall back to brute-force if ay fails
    ForceAY,
}

// Part of #2757: ay-specific feature flags and methods are gated behind
// cfg(feature = "ay"). All production callers (init_solve.rs) are already
// behind this gate; gating the definitions prevents dead code when ay is
// disabled, rather than relying on test usage to mask the warnings.
#[cfg(feature = "ay")]
feature_flag!(force_ay_enabled, "TY_FORCE_AY");
#[cfg(feature = "ay")]
feature_flag!(auto_ay_enabled, "TY_USE_AY");

#[cfg(feature = "ay")]
impl InitMode {
    /// Resolve effective mode from env vars and config.
    ///
    /// Priority order:
    /// 1. TY_FORCE_AY env var → ForceAY (always try ay)
    /// 2. TY_USE_AY env var → Auto (use ay if analysis recommends)
    /// 3. config_mode value (if no env vars set)
    ///
    /// Note: TY_USE_AY forces Auto mode regardless of config_mode.
    /// This means if config_mode is ForceAY, setting TY_USE_AY will
    /// downgrade to Auto. Use TY_FORCE_AY to override to ForceAY.
    pub fn resolve(config_mode: InitMode) -> InitMode {
        if force_ay_enabled() {
            InitMode::ForceAY
        } else if auto_ay_enabled() {
            InitMode::Auto
        } else {
            config_mode
        }
    }

    /// Given analysis needs_ay result, determine (force_ay, should_try_ay) tuple.
    ///
    /// - `force_ay`: Whether to force ay even if analysis doesn't recommend it
    /// - `should_try_ay`: Whether to attempt ay-based enumeration
    pub fn ay_decision(self, analysis_needs_ay: bool) -> (bool, bool) {
        match self {
            InitMode::BruteForce => (false, false),
            InitMode::Auto => (false, analysis_needs_ay),
            InitMode::ForceAY => (true, true),
        }
    }

    /// Returns true if analysis should be skipped (for optimization).
    ///
    /// When BruteForce mode is set, we can skip expensive analysis.
    pub fn should_skip_analysis(self) -> bool {
        matches!(self, InitMode::BruteForce)
    }
}
