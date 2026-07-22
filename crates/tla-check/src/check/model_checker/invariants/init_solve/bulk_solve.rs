// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Streaming bulk init-state solving: `solve_predicate_for_states_to_bulk`.

use super::super::super::{BulkStateStorage, CheckError, ModelChecker, Value};
use super::{BulkInitStates, Constraint, EvalCheckError};
#[cfg(feature = "ay")]
use crate::enumerate::BulkConstraintEnumerationStats;
use crate::enumerate::{
    enumerate_constraints_to_bulk_with_stats,
    enumerate_constraints_to_bulk_with_stats_filter_error, eval_filter_expr,
    BulkConstraintEnumerationError,
};

impl<'a> ModelChecker<'a> {
    /// Solve a predicate and stream results directly to BulkStateStorage.
    ///
    /// Memory-efficient alternative to `solve_predicate_for_states` — avoids intermediate
    /// `State` (OrdMap) objects. For MCBakery ISpec with 655K states this eliminates 655K
    /// OrdMap allocations.
    ///
    /// Returns `Ok(None)` when streaming is not possible (caller falls back to Vec path).
    pub(in crate::check::model_checker) fn solve_predicate_for_states_to_bulk(
        &mut self,
        pred_name: &str,
    ) -> Result<Option<BulkInitStates>, CheckError> {
        let resolved = self.resolve_init_predicate(pred_name)?;

        // Try ay enumeration first (skip analysis overhead in BruteForce mode).
        #[cfg(feature = "ay")]
        if let Some(states) = self.try_ay_init_states(&resolved, false) {
            let registry = self.ctx.var_registry();
            let mut storage = BulkStateStorage::new(registry.len(), states.len());
            for state in states {
                storage.push_state_iter(Vec::from(state.to_values(registry)));
            }
            let distinct = storage.len();
            return Ok(Some(BulkInitStates {
                storage,
                enumeration: BulkConstraintEnumerationStats {
                    generated: distinct,
                    added: distinct,
                },
            }));
        }

        // Direct constraint enumeration to bulk.
        if let Some(branches) = &resolved.extracted_branches {
            if resolved.unconstrained_vars.is_empty() {
                let vars_len = self.ctx.var_registry().len();
                let mut storage = BulkStateStorage::new(vars_len, 1000);
                let count = enumerate_constraints_to_bulk_with_stats(
                    &mut self.ctx,
                    &self.module.vars,
                    branches,
                    &mut storage,
                    |_values, _ctx| Ok(true),
                );
                return match count {
                    Ok(Some(stats)) => Ok(Some(BulkInitStates {
                        storage,
                        enumeration: stats,
                    })),
                    Ok(None) => Ok(None),
                    Err(e) => Err(EvalCheckError::Eval(e).into()),
                };
            }
        }

        // Fallback: stream from a type predicate with canonical AST filtering.
        let pred_body = &self.module.op_defs[&resolved.resolved_name].body;
        let candidates = self.find_type_candidates(pred_name);
        for cand_name in candidates {
            let Some(branches) = self.candidate_branches(&cand_name) else {
                continue;
            };
            let filter_expr = self.candidate_remainder_filter_expr(pred_body, &cand_name);

            let vars_len = self.ctx.var_registry().len();
            let mut storage = BulkStateStorage::new(vars_len, 1000);
            let count = enumerate_constraints_to_bulk_with_stats(
                &mut self.ctx,
                &self.module.vars,
                &branches,
                &mut storage,
                |_values, ctx| match filter_expr.as_ref() {
                    Some(expr) => eval_filter_expr(ctx, expr),
                    None => Ok(true),
                },
            );
            match count {
                Ok(Some(stats)) => {
                    return Ok(Some(BulkInitStates {
                        storage,
                        enumeration: stats,
                    }));
                }
                Ok(None) => {}
                Err(e) => return Err(EvalCheckError::Eval(e).into()),
            }
        }

        Ok(None)
    }

    /// Try to generate simulation initial states directly into bulk storage.
    ///
    /// This uses the normal streaming solver first. If that cannot enumerate
    /// because simulation has unconstrained variables, it applies the simulation
    /// default of integer 0 for those variables and streams the augmented
    /// constraints into `BulkStateStorage`.
    pub(in crate::check::model_checker) fn generate_initial_states_simulation_to_bulk(
        &mut self,
        init_name: &str,
    ) -> Result<Option<BulkInitStates>, CheckError> {
        if let Some(states) = self.solve_predicate_for_states_to_bulk(init_name)? {
            return Ok(Some(states));
        }

        let resolved = self.resolve_init_predicate(init_name)?;
        let Some(mut branches) = resolved.extracted_branches else {
            return Ok(None);
        };
        if resolved.unconstrained_vars.is_empty() {
            return Ok(None);
        }

        for var_name in &resolved.unconstrained_vars {
            for branch in &mut branches {
                branch.push(Constraint::Eq(var_name.clone(), Value::int(0)));
            }
        }

        let pred_body = &self.module.op_defs[&resolved.resolved_name].body;
        let vars_len = self.ctx.var_registry().len();
        let mut storage = BulkStateStorage::new(vars_len, 1000);
        let count = enumerate_constraints_to_bulk_with_stats_filter_error(
            &mut self.ctx,
            &self.module.vars,
            &branches,
            &mut storage,
            |_values, ctx| match crate::error_policy::eval_speculative(ctx, pred_body, &[])? {
                Some(Value::Bool(keep)) => Ok(keep),
                Some(_) => Err(EvalCheckError::InitNotBoolean.into()),
                None => unreachable!("empty fallback class list cannot suppress eval errors"),
            },
        );

        match count {
            Ok(Some(enumeration)) => Ok(Some(BulkInitStates {
                storage,
                enumeration,
            })),
            Ok(None) => Ok(None),
            Err(BulkConstraintEnumerationError::Eval(e)) => Err(EvalCheckError::Eval(e).into()),
            Err(BulkConstraintEnumerationError::Filter(e)) => Err(e),
        }
    }
}
