// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! The compiled-BFS admission gate, lifted out of `tla-check` so the decision lives in
//! one auditable place (Stage 4 of the unified-backend migration).
//!
//! [`admit_compiled_bfs`] is a faithful, line-for-line reproduction of
//! `ModelChecker::should_use_compiled_bfs` (`crates/tla-check/src/check/model_checker/
//! run_helpers.rs`). `tla-check` implements [`CompiledBfsFacts`] by pure delegation to
//! its existing methods and now *delegates the decision here*, keeping the original body
//! as a `debug_assert_eq!` shadow oracle so any future divergence is caught in debug
//! builds and the differential `supremacy` sweep.
//!
//! Fail-closed: every early return is `false` (interpreter); a native verdict is only
//! reached by passing every gate.

/// The exact facts the compiled-BFS gate reads. Each maps 1:1 to an existing
/// `ModelChecker` method or `Config` field, so the consumer impl is trivial delegation.
pub trait CompiledBfsFacts {
    /// `config.use_compiled_bfs` — programmatic force enable (`Some(true)`) / disable
    /// (`Some(false)`) / auto (`None`).
    fn use_compiled_bfs_override(&self) -> Option<bool>;
    /// `TY_NO_COMPILED_BFS` env force-disable.
    fn compiled_bfs_env_disabled(&self) -> bool;
    /// The flat frontier layout is admitted for the compiled loop.
    fn flat_frontier_admitted(&self) -> bool;
    /// The compiled step width matches the flat frontier.
    fn step_width_matches_flat_frontier(&self) -> bool;
    /// The spec has implied actions that require interpreter evaluation.
    fn implied_actions_require_interpreter_eval(&self) -> bool;
    /// The installed per-parent STEP path evaluates implied actions via the interpreter
    /// hook per edge (the exception that keeps the compiled loop admissible).
    fn step_evaluates_interpreter_implied_actions(&self) -> bool;
    /// `!config.action_constraints.is_empty()`.
    fn has_action_constraints(&self) -> bool;
    /// Per-action coverage attribution is active. The compiled loop does not
    /// emit action counts and must yield to the split-action dispatcher.
    fn coverage_collect(&self) -> bool;
    /// `!config.constraints.is_empty()` (state constraints).
    fn has_state_constraints(&self) -> bool;
    /// Under state constraints, whether the native-fused admission is active.
    fn state_constrained_native_fused_admission_active(&self) -> bool;
    /// A compiled BFS step is built (`compiled_bfs_step.is_some()`).
    fn compiled_step_built(&self) -> bool;
    /// A compiled BFS fused level is built (`compiled_bfs_level.is_some()`).
    fn compiled_level_built(&self) -> bool;
    /// The state layout is fully flat (all-scalar).
    fn fully_flat_layout(&self) -> bool;
}

/// Whether the compiled BFS loop is admitted for this run. Faithful reproduction of
/// `ModelChecker::should_use_compiled_bfs`; the step numbers match its comments.
#[must_use]
pub fn admit_compiled_bfs(f: &dyn CompiledBfsFacts) -> bool {
    // 1. Programmatic force-disable
    if f.use_compiled_bfs_override() == Some(false) {
        return false;
    }
    // 2. Env var force-disable
    if f.compiled_bfs_env_disabled() {
        return false;
    }
    if !f.flat_frontier_admitted() {
        return false;
    }
    if !f.step_width_matches_flat_frontier() {
        return false;
    }
    if f.implied_actions_require_interpreter_eval()
        && !f.step_evaluates_interpreter_implied_actions()
    {
        // Non-native implied actions normally fence off compiled BFS. The exception:
        // when only the per-parent STEP path is installed (no fused LEVEL) and it
        // preserves every successor edge, the compiled loop evaluates the implied
        // action per edge via the interpreter hook, so it is still admissible.
        return false;
    }
    if f.has_action_constraints() {
        return false;
    }
    if f.coverage_collect() {
        return false;
    }
    if f.has_state_constraints() {
        return f.state_constrained_native_fused_admission_active();
    }
    // 3. Programmatic force-enable (if compiled step or fused level is ready)
    if f.use_compiled_bfs_override() == Some(true) {
        return f.compiled_step_built() || f.compiled_level_built();
    }
    // 4. Auto-detect for all-scalar specs: a compiled step or fused level is built AND
    // the state layout is fully flat (no compound types).
    if !f.compiled_step_built() && !f.compiled_level_built() {
        return false;
    }
    // Verify the state layout is fully flat (all-scalar).
    f.fully_flat_layout()
}

#[cfg(test)]
mod tests {
    use super::{admit_compiled_bfs, CompiledBfsFacts};

    #[derive(Clone, Copy)]
    struct Facts {
        override_: Option<bool>,
        env_disabled: bool,
        flat_frontier_admitted: bool,
        step_width_matches: bool,
        implied_require_interp: bool,
        step_evaluates_implied: bool,
        action_constraints: bool,
        coverage_collect: bool,
        state_constraints: bool,
        state_constrained_native_fused: bool,
        step_built: bool,
        level_built: bool,
        fully_flat: bool,
    }

    /// A fully-admissible auto run: built step + fully flat, no fences.
    impl Default for Facts {
        fn default() -> Self {
            Facts {
                override_: None,
                env_disabled: false,
                flat_frontier_admitted: true,
                step_width_matches: true,
                implied_require_interp: false,
                step_evaluates_implied: false,
                action_constraints: false,
                coverage_collect: false,
                state_constraints: false,
                state_constrained_native_fused: false,
                step_built: true,
                level_built: false,
                fully_flat: true,
            }
        }
    }

    impl CompiledBfsFacts for Facts {
        fn use_compiled_bfs_override(&self) -> Option<bool> {
            self.override_
        }
        fn compiled_bfs_env_disabled(&self) -> bool {
            self.env_disabled
        }
        fn flat_frontier_admitted(&self) -> bool {
            self.flat_frontier_admitted
        }
        fn step_width_matches_flat_frontier(&self) -> bool {
            self.step_width_matches
        }
        fn implied_actions_require_interpreter_eval(&self) -> bool {
            self.implied_require_interp
        }
        fn step_evaluates_interpreter_implied_actions(&self) -> bool {
            self.step_evaluates_implied
        }
        fn has_action_constraints(&self) -> bool {
            self.action_constraints
        }
        fn coverage_collect(&self) -> bool {
            self.coverage_collect
        }
        fn has_state_constraints(&self) -> bool {
            self.state_constraints
        }
        fn state_constrained_native_fused_admission_active(&self) -> bool {
            self.state_constrained_native_fused
        }
        fn compiled_step_built(&self) -> bool {
            self.step_built
        }
        fn compiled_level_built(&self) -> bool {
            self.level_built
        }
        fn fully_flat_layout(&self) -> bool {
            self.fully_flat
        }
    }

    #[test]
    fn admissible_auto_run_is_admitted() {
        assert!(admit_compiled_bfs(&Facts::default()));
    }

    #[test]
    fn action_coverage_vetoes_compiled_bfs() {
        assert!(!admit_compiled_bfs(&Facts {
            coverage_collect: true,
            override_: Some(true),
            ..Facts::default()
        }));
    }

    #[test]
    fn force_disable_wins_over_everything() {
        let f = Facts {
            override_: Some(false),
            ..Facts::default()
        };
        assert!(!admit_compiled_bfs(&f));
    }

    #[test]
    fn env_disable_fences() {
        assert!(!admit_compiled_bfs(&Facts {
            env_disabled: true,
            ..Facts::default()
        }));
    }

    #[test]
    fn frontier_and_width_gates_fence() {
        assert!(!admit_compiled_bfs(&Facts {
            flat_frontier_admitted: false,
            ..Facts::default()
        }));
        assert!(!admit_compiled_bfs(&Facts {
            step_width_matches: false,
            ..Facts::default()
        }));
    }

    #[test]
    fn implied_actions_fence_unless_step_evaluates_them() {
        // require interp + step does NOT evaluate -> fenced
        assert!(!admit_compiled_bfs(&Facts {
            implied_require_interp: true,
            step_evaluates_implied: false,
            ..Facts::default()
        }));
        // require interp + step DOES evaluate -> admissible
        assert!(admit_compiled_bfs(&Facts {
            implied_require_interp: true,
            step_evaluates_implied: true,
            ..Facts::default()
        }));
    }

    #[test]
    fn action_constraints_fence() {
        assert!(!admit_compiled_bfs(&Facts {
            action_constraints: true,
            ..Facts::default()
        }));
    }

    #[test]
    fn state_constraints_defer_to_native_fused_admission() {
        // state constraints + native-fused active -> admitted
        assert!(admit_compiled_bfs(&Facts {
            state_constraints: true,
            state_constrained_native_fused: true,
            ..Facts::default()
        }));
        // state constraints + native-fused inactive -> NOT admitted
        assert!(!admit_compiled_bfs(&Facts {
            state_constraints: true,
            state_constrained_native_fused: false,
            ..Facts::default()
        }));
    }

    #[test]
    fn force_enable_requires_a_built_artifact() {
        // Some(true) + a built step -> admitted
        assert!(admit_compiled_bfs(&Facts {
            override_: Some(true),
            step_built: true,
            level_built: false,
            fully_flat: false, // force-enable bypasses the fully-flat auto-detect
            ..Facts::default()
        }));
        // Some(true) + nothing built -> NOT admitted
        assert!(!admit_compiled_bfs(&Facts {
            override_: Some(true),
            step_built: false,
            level_built: false,
            ..Facts::default()
        }));
    }

    #[test]
    fn auto_requires_built_artifact_and_fully_flat() {
        // nothing built -> not admitted
        assert!(!admit_compiled_bfs(&Facts {
            step_built: false,
            level_built: false,
            ..Facts::default()
        }));
        // built but not fully flat -> not admitted
        assert!(!admit_compiled_bfs(&Facts {
            fully_flat: false,
            ..Facts::default()
        }));
        // level built + fully flat -> admitted
        assert!(admit_compiled_bfs(&Facts {
            step_built: false,
            level_built: true,
            fully_flat: true,
            ..Facts::default()
        }));
    }
}
