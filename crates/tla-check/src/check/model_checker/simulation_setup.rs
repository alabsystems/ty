// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared setup helpers for simulation-mode execution.

use super::simulation_types::{SimulationConfig, SimulationStats};
use super::{
    bind_constants_from_config, build_ident_hints, precompute_constant_operators,
    promote_env_constants_to_precomputed, ArrayState, BulkInitStates, BulkStateStorage, CheckError,
    CheckResult, ModelChecker, State,
};
use crate::eval::TlcRuntimeStats;
use crate::{ConfigCheckError, EvalCheckError};

pub(super) enum SimulationInitialStates {
    Vec(Vec<State>),
    Bulk {
        storage: BulkStateStorage,
        accepted_indices: Option<Vec<u32>>,
    },
}

impl SimulationInitialStates {
    #[inline]
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Vec(states) => states.len(),
            Self::Bulk {
                storage,
                accepted_indices,
            } => accepted_indices
                .as_ref()
                .map_or_else(|| storage.len(), Vec::len),
        }
    }

    #[inline]
    pub(super) fn array_state_at(
        &self,
        idx: usize,
        registry: &crate::var_index::VarRegistry,
    ) -> ArrayState {
        match self {
            Self::Vec(states) => ArrayState::from_state_with_fp(&states[idx], registry),
            Self::Bulk {
                storage,
                accepted_indices,
            } => ArrayState::from_values(
                storage
                    .get_state(bulk_storage_index(idx, accepted_indices.as_deref()))
                    .to_vec(),
            ),
        }
    }

    #[inline]
    pub(super) fn state_at(&self, idx: usize, registry: &crate::var_index::VarRegistry) -> State {
        match self {
            Self::Vec(states) => states[idx].clone(),
            Self::Bulk {
                storage,
                accepted_indices,
            } => State::from_indexed(
                storage.get_state(bulk_storage_index(idx, accepted_indices.as_deref())),
                registry,
            ),
        }
    }
}

#[inline]
fn bulk_storage_index(pool_idx: usize, accepted_indices: Option<&[u32]>) -> u32 {
    accepted_indices.map_or_else(
        || u32::try_from(pool_idx).expect("BulkStateStorage state index does not fit in u32"),
        |indices| indices[pool_idx],
    )
}

impl ModelChecker<'_> {
    pub(super) fn update_simulation_runtime_stats(
        &self,
        stats: &SimulationStats,
        distinct_states: usize,
        current_trace: usize,
    ) {
        self.ctx.set_tlc_runtime_stats(Some(TlcRuntimeStats::new(
            stats.states_visited as i64,
            distinct_states as i64,
            0,
            current_trace as i64,
            current_trace as i64,
        )));
    }

    pub(super) fn prepare_simulation(
        &mut self,
        sim_config: &SimulationConfig,
    ) -> Result<(SimulationInitialStates, String), CheckError> {
        self.sync_simulation_tlc_config(sim_config);
        self.prepare_simulation_constants()?;
        self.validate_simulation_model()?;
        let (init_name, next_name) = self.resolve_simulation_operator_names()?;
        self.ensure_simulation_assumes_hold()?;
        self.detect_simulation_trace_usage()?;
        let initial_states = self.generate_constrained_initial_state_pool(&init_name)?;
        Ok((initial_states, next_name))
    }

    fn sync_simulation_tlc_config(&mut self, sim_config: &SimulationConfig) {
        // Sync TLC config for TLCGet("config") support - use "generate" mode for simulation
        self.sync_tlc_config("generate");
        // Part of #3076: propagate simulation-specific config to TLCGet("config")
        let mut tlc_config = self.ctx.shared().tlc_config.clone();
        tlc_config.traces = sim_config.num_traces as i64;
        tlc_config.seed = sim_config.seed.unwrap_or(0) as i64;
        self.ctx.set_tlc_config(tlc_config);
    }

    fn prepare_simulation_constants(&mut self) -> Result<(), CheckError> {
        // Bind constants from config before checking.
        bind_constants_from_config(&mut self.ctx, self.config)
            .map_err(|error| CheckError::from(EvalCheckError::Eval(error)))?;
        // Part of #3078: Must match prepare_bfs_common setup — without these,
        // eval_ident's fast path skips env.get() for interned names when state_env
        // is set, causing "Undefined variable" errors for constants like N.
        // See run_gen.rs:47-48 for the same pattern in the BFS pilot path.
        precompute_constant_operators(&mut self.ctx);
        promote_env_constants_to_precomputed(&mut self.ctx);
        build_ident_hints(&mut self.ctx);
        self.invariant_verdict_cache
            .rebuild(&self.ctx, &self.config.invariants);
        self.state_constraint_verdict_cache
            .rebuild(&self.ctx, &self.config.constraints);
        Ok(())
    }

    fn validate_simulation_model(&self) -> Result<(), CheckError> {
        if self.module.vars.is_empty() {
            return Err(ConfigCheckError::NoVariables.into());
        }
        for inv_name in &self.config.invariants {
            if !self.ctx.has_op(inv_name) {
                return Err(ConfigCheckError::MissingInvariant(inv_name.clone()).into());
            }
        }
        Ok(())
    }

    fn resolve_simulation_operator_names(&self) -> Result<(String, String), CheckError> {
        let init_name = self
            .config
            .init
            .as_ref()
            .ok_or(ConfigCheckError::MissingInit)?;
        // Part of #3078: Apply CONSTANT operator replacements (e.g., `Init <- SimInit`).
        // BFS does this at run_gen.rs:50; simulation was missing it, causing the original
        // operator body to be used instead of the replacement.
        let init_name = self.ctx.resolve_op_name(init_name).to_string();

        let next_name = self
            .config
            .next
            .as_ref()
            .ok_or(ConfigCheckError::MissingNext)?;
        let next_name = self.ctx.resolve_op_name(next_name).to_string();

        Ok((init_name, next_name))
    }

    fn ensure_simulation_assumes_hold(&self) -> Result<(), CheckError> {
        // Check ASSUME statements before state exploration (matches TLC semantics).
        // Part of #3076: simulation path was missing this check, causing assume_violation
        // errors to only appear when diagnose ran specs under `ty check` instead of simulate.
        match self.check_assumes() {
            Some(result) => Err(assume_failure_to_error(result)),
            None => Ok(()),
        }
    }

    fn detect_simulation_trace_usage(&mut self) -> Result<(), CheckError> {
        // Part of #1121: Shared conservative Trace detector across checker paths.
        self.compiled.uses_trace =
            super::trace_detect::compute_uses_trace(self.config, &self.module.op_defs)?;
        Ok(())
    }

    fn generate_constrained_initial_state_pool(
        &mut self,
        init_name: &str,
    ) -> Result<SimulationInitialStates, CheckError> {
        if let Some(bulk) = self.generate_initial_states_simulation_to_bulk(init_name)? {
            return self.filter_bulk_initial_states(bulk);
        }

        let initial_states = self.generate_constrained_initial_states_vec(init_name)?;
        Ok(SimulationInitialStates::Vec(initial_states))
    }

    fn generate_constrained_initial_states_vec(
        &mut self,
        init_name: &str,
    ) -> Result<Vec<State>, CheckError> {
        let initial_states = self.generate_initial_states_simulation_fallback(init_name)?;
        let registry = self.ctx.var_registry().clone();
        let mut constrained_initial_states = Vec::with_capacity(initial_states.len());
        for state in initial_states {
            let arr = ArrayState::from_state(&state, &registry);
            if self.check_state_constraints_array(&arr)? {
                constrained_initial_states.push(state);
            }
        }
        if constrained_initial_states.is_empty() {
            return Err(ConfigCheckError::InitCannotEnumerate(
                "No valid initial states after constraint filtering".to_string(),
            )
            .into());
        }
        Ok(constrained_initial_states)
    }

    fn filter_bulk_initial_states(
        &mut self,
        bulk: BulkInitStates,
    ) -> Result<SimulationInitialStates, CheckError> {
        let BulkInitStates { storage, .. } = bulk;
        if storage.is_empty() {
            return Err(no_simulation_initial_states_error());
        }
        if self.config.constraints.is_empty() {
            return Ok(SimulationInitialStates::Bulk {
                storage,
                accepted_indices: None,
            });
        }

        let registry = self.ctx.var_registry().clone();
        let mut scratch = ArrayState::new(registry.len());
        let mut accepted_indices = Vec::new();
        for idx in 0..storage.len() {
            let idx = u32::try_from(idx).map_err(|_| {
                ConfigCheckError::Setup(format!(
                    "too many simulation initial states ({}) for u32 BulkStateStorage index",
                    storage.len()
                ))
            })?;
            scratch.overwrite_from_slice(storage.get_state(idx));
            if self.check_state_constraints_array(&scratch)? {
                accepted_indices.push(idx);
            }
        }
        if accepted_indices.is_empty() {
            return Err(no_simulation_initial_states_error());
        }

        let accepted_indices =
            (accepted_indices.len() != storage.len()).then_some(accepted_indices);
        Ok(SimulationInitialStates::Bulk {
            storage,
            accepted_indices,
        })
    }
}

fn assume_failure_to_error(result: CheckResult) -> CheckError {
    match result {
        CheckResult::Error { error, .. } => error,
        other => ConfigCheckError::Setup(format!(
            "ASSUME check returned unexpected result: {:?}",
            other.stats()
        ))
        .into(),
    }
}

fn no_simulation_initial_states_error() -> CheckError {
    ConfigCheckError::InitCannotEnumerate(
        "No valid initial states after constraint filtering".to_string(),
    )
    .into()
}

#[cfg(test)]
mod tests {
    use super::super::simulation_types::SimulationResult;
    use super::*;
    use crate::config::{Config, ConstantValue};
    use crate::Value;
    use tla_core::{lower, parse_to_syntax_tree, FileId};

    fn parse_module(src: &str) -> tla_core::ast::Module {
        let tree = parse_to_syntax_tree(src);
        lower(FileId(0), &tree).module.unwrap()
    }

    fn sim_config() -> SimulationConfig {
        SimulationConfig {
            num_traces: 1,
            max_trace_length: 1,
            seed: Some(7),
            check_invariants: false,
            action_constraints: Vec::new(),
        }
    }

    #[test]
    fn simulation_initial_pool_uses_bulk_for_enumerable_init() {
        let module = parse_module(
            r#"
---- MODULE SimBulkInit ----
VARIABLE x
Init == x \in {0, 1, 2, 3}
Next == x' = x
===="#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);

        let (pool, _) = checker.prepare_simulation(&sim_config()).unwrap();

        assert_eq!(pool.len(), 4);
        assert!(matches!(pool, SimulationInitialStates::Bulk { .. }));
    }

    #[test]
    fn simulation_bulk_initial_pool_filters_by_random_index() {
        let module = parse_module(
            r#"
---- MODULE SimBulkInitConstraint ----
VARIABLE x
Init == x \in {0, 1, 2, 3}
Next == x' = x
Good == x = 0 \/ x = 2
===="#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            constraints: vec!["Good".to_string()],
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);

        let (pool, _) = checker.prepare_simulation(&sim_config()).unwrap();
        let registry = checker.ctx.var_registry().clone();
        let values = (0..pool.len())
            .map(|idx| pool.state_at(idx, &registry).get("x").unwrap().clone())
            .collect::<Vec<_>>();

        assert_eq!(pool.len(), 2);
        assert!(matches!(
            pool,
            SimulationInitialStates::Bulk {
                accepted_indices: Some(_),
                ..
            }
        ));
        assert_eq!(values, vec![Value::int(0), Value::int(2)]);
    }

    #[test]
    fn simulation_bulk_initial_pool_applies_unconstrained_defaults() {
        let module = parse_module(
            r#"
---- MODULE SimBulkInitDefault ----
VARIABLE x, y
Init == x \in {1, 2}
Next == x' = x /\ y' = y
===="#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);

        let (pool, _) = checker.prepare_simulation(&sim_config()).unwrap();
        let registry = checker.ctx.var_registry().clone();
        let y_values = (0..pool.len())
            .map(|idx| pool.state_at(idx, &registry).get("y").unwrap().clone())
            .collect::<Vec<_>>();

        assert_eq!(pool.len(), 2);
        assert!(matches!(pool, SimulationInitialStates::Bulk { .. }));
        assert_eq!(y_values, vec![Value::int(0), Value::int(0)]);
    }

    #[test]
    fn simulation_smoke_init_override_defers_opaque_extends_filter_until_state_bound() {
        let base = parse_module(
            r#"
---- MODULE SimSmokeBase ----
VARIABLE pending, counter
vars == <<pending, counter>>

Init == pending = 0 /\ counter = 0
Next == UNCHANGED vars

Inv ==
  /\ P0:: pending = 0
  /\ TRUE
===="#,
        );
        let root = parse_module(
            r#"
---- MODULE SimSmokeRoot ----
EXTENDS SimSmokeBase

SmokeInit ==
  /\ pending \in {0, 1}
  /\ counter \in {0, 1}
  /\ Inv!P0
===="#,
        );
        let mut config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        config.add_constant(
            "Init".to_string(),
            ConstantValue::Replacement("SmokeInit".to_string()),
        );

        let mut checker = ModelChecker::new_with_extends(&root, &[&base], &config);
        let (pool, _) = checker
            .prepare_simulation(&sim_config())
            .expect("Smoke-style Init override should enumerate with opaque label filter");
        let registry = checker.ctx.var_registry().clone();
        let pending_idx = registry.get("pending").expect("pending var");
        let counter_idx = registry.get("counter").expect("counter var");

        let mut values = (0..pool.len())
            .map(|idx| {
                let state = pool.array_state_at(idx, &registry);
                (state.get(pending_idx), state.get(counter_idx))
            })
            .collect::<Vec<_>>();
        values.sort();

        assert_eq!(
            values,
            vec![
                (Value::int(0), Value::int(0)),
                (Value::int(0), Value::int(1)),
            ],
            "opaque Inv!P0 filter must run after imported state variables are bound"
        );
    }

    #[test]
    fn simulation_bulk_initial_pool_runs_simulation_loop() {
        let module = parse_module(
            r#"
---- MODULE SimBulkInitLoop ----
EXTENDS Naturals
VARIABLE x, y
Init == x \in {0, 1, 2} /\ y = 0
Next == x' = x + 1 /\ y' = y
Inv == y = 0
===="#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Inv".to_string()],
            check_deadlock: false,
            ..Default::default()
        };
        let sim_config = SimulationConfig {
            num_traces: 4,
            max_trace_length: 2,
            seed: Some(11),
            check_invariants: true,
            action_constraints: Vec::new(),
        };
        let mut checker = ModelChecker::new(&module, &config);

        let result = checker.simulate(&sim_config);

        match result {
            SimulationResult::Success(stats) => {
                assert_eq!(stats.traces_generated, 4);
                assert_eq!(stats.states_visited, 12);
                assert_eq!(stats.transitions, 8);
                assert_eq!(stats.truncated_traces, 4);
                assert!(stats.distinct_states >= 3);
            }
            other => panic!("expected simulation success, got {other:?}"),
        }
    }
}
