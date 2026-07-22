// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared init-predicate resolution and constraint extraction helpers.

#[cfg(all(debug_assertions, feature = "ay"))]
use super::super::super::debug::debug_init;
#[cfg(feature = "ay")]
use super::super::super::InitMode;
use super::super::super::{
    extract_conjunction_remainder, extract_init_constraints, find_unconstrained_vars, ModelChecker,
};
#[cfg(feature = "ay")]
use super::State;
use super::{CheckError, ConfigCheckError, Constraint, InitPredResolved};
#[cfg(feature = "ay")]
use crate::enumerate::collect_state_var_refs;
#[cfg(feature = "ay")]
use crate::init_strategy::analyze_init_predicate;

#[cfg(feature = "ay")]
fn init_omits_any_unconstrained_var(
    ctx: &crate::eval::EvalCtx,
    init_expr: &tla_core::Spanned<tla_core::ast::Expr>,
    vars: &[std::sync::Arc<str>],
    unconstrained_vars: &[String],
) -> bool {
    if unconstrained_vars.is_empty() {
        return false;
    }

    let mut referenced = rustc_hash::FxHashSet::default();
    collect_state_var_refs(ctx, init_expr, vars, &mut referenced);
    let referenced_names: std::collections::HashSet<&str> =
        referenced.iter().map(std::sync::Arc::as_ref).collect();
    unconstrained_vars
        .iter()
        .any(|var| !referenced_names.contains(var.as_str()))
}

impl<'a> ModelChecker<'a> {
    // ── Shared helpers ──────────────────────────────────────────────

    /// Resolve the predicate name, get its definition, and extract constraints.
    pub(super) fn resolve_init_predicate(
        &self,
        pred_name: &str,
    ) -> Result<InitPredResolved, CheckError> {
        let resolved_name = self.ctx.resolve_op_name(pred_name).to_string();
        let def = self
            .module
            .op_defs
            .get(&resolved_name)
            .ok_or(ConfigCheckError::MissingInit)?;
        // Part of #3296: TIR re-enabled for init constraint extraction.
        // The binding-form gate in try_eval_expr_detailed() catches FuncDef/
        // SetFilter/SetBuilder, making the prior blanket None gate redundant.
        // Verified: YoYoAllGraphs (26731 states) and MCtcp (1182 states) pass.
        let tir_program = self.tir_parity.as_ref().and_then(|tir| {
            tir.make_tir_program_for_selected_leaf_eval_name(pred_name, &resolved_name)
        });
        let extraction_result = extract_init_constraints(
            &self.ctx,
            &def.body,
            &self.module.vars,
            tir_program.as_ref(),
        );
        let constraint_extraction_succeeded = extraction_result.is_some();
        let (extracted_branches, unconstrained_vars) = match extraction_result {
            Some(branches) => {
                let unconstrained = find_unconstrained_vars(&self.module.vars, &branches);
                (Some(branches), unconstrained)
            }
            None => (None, Vec::new()),
        };
        Ok(InitPredResolved {
            resolved_name,
            extracted_branches,
            unconstrained_vars,
            constraint_extraction_succeeded,
        })
    }

    /// Try ay-based SMT enumeration. Returns `Some(states)` on success, `None` on failure/skip.
    ///
    /// If `always_analyze` is true, always runs `analyze_init_predicate` (needed for the
    /// Vec path which uses the analysis in error messages). If false, skips analysis when
    /// the init mode is BruteForce (an optimisation the bulk path uses).
    #[cfg(feature = "ay")]
    pub(super) fn try_ay_init_states(
        &self,
        resolved: &InitPredResolved,
        always_analyze: bool,
    ) -> Option<Vec<State>> {
        let def = self.module.op_defs.get(&resolved.resolved_name)?;
        if init_omits_any_unconstrained_var(
            &self.ctx,
            &def.body,
            &self.module.vars,
            &resolved.unconstrained_vars,
        ) {
            debug_eprintln!(
                debug_init(),
                "[init-strategy] skipping ay enumeration: Init omits unconstrained state variable(s)"
            );
            return None;
        }
        let effective_mode = InitMode::resolve(self.config.init_mode);
        let needs_ay = if !always_analyze && effective_mode.should_skip_analysis() {
            false
        } else {
            analyze_init_predicate(
                &def.body,
                &self.module.vars,
                resolved.constraint_extraction_succeeded,
                &resolved.unconstrained_vars,
            )
            .needs_ay()
        };
        let (force_ay, should_try_ay) = effective_mode.ay_decision(needs_ay);
        #[cfg(not(debug_assertions))]
        let _ = force_ay;
        debug_block!(debug_init(), {
            eprintln!(
                "[init-strategy] ay check: force={}, needs_ay={}, should_try={}",
                force_ay, needs_ay, should_try_ay
            );
        });
        if !should_try_ay {
            return None;
        }

        // Part of #3826: Try specialized nested powerset encoder first.
        // This handles patterns like SUBSET(SUBSET Nodes) with cardinality
        // filters much more efficiently than the general ay translator.
        if let Some(info) = crate::ay_init_powerset::detect_nested_powerset_init(
            &self.ctx,
            &def.body,
            &self.module.vars,
        ) {
            debug_eprintln!(
                debug_init(),
                "[init-strategy] Detected nested powerset pattern for '{}', \
                 using specialized encoder ({} base elements)",
                info.var_name,
                info.base_elements.len()
            );
            match crate::ay_init_powerset::enumerate_nested_powerset_init(&info) {
                Ok(states) => {
                    debug_eprintln!(
                        debug_init(),
                        "[init-strategy] Nested powerset enumeration succeeded: {} states",
                        states.len()
                    );
                    return Some(states);
                }
                Err(e) => {
                    debug_eprintln!(
                        debug_init(),
                        "[init-strategy] Nested powerset enumeration failed: {}, \
                         falling back to general ay",
                        e
                    );
                    // Fall through to general ay enumeration
                }
            }
        }

        use crate::ay_enumerate::{enumerate_init_states_ay, AYEnumConfig};
        debug_eprintln!(debug_init(), "[init-strategy] Attempting ay enumeration...");
        let config = AYEnumConfig::default();
        match enumerate_init_states_ay(&self.ctx, &def.body, &self.module.vars, None, &config) {
            Ok(states) => {
                debug_eprintln!(
                    debug_init(),
                    "[init-strategy] ay enumeration succeeded: {} states",
                    states.len()
                );
                Some(states)
            }
            Err(e) => {
                debug_eprintln!(
                    debug_init(),
                    "[init-strategy] ay enumeration failed: {}, falling back",
                    e
                );
                None
            }
        }
    }

    /// Collect candidate type-predicates (TypeOK, configured invariants) for fallback enumeration.
    pub(super) fn find_type_candidates(&self, pred_name: &str) -> Vec<String> {
        let mut candidates: Vec<String> = Vec::new();
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for name in ["TypeOK", "TypeOk"] {
            if name != pred_name && seen.insert(name) {
                candidates.push(name.to_string());
            }
        }
        for inv in &self.config.invariants {
            let inv_name = inv.as_str();
            if inv_name != pred_name && seen.insert(inv_name) {
                candidates.push(inv.clone());
            }
        }
        candidates
    }

    /// For a candidate name, extract its constraint branches if fully constrained.
    pub(super) fn candidate_branches(&self, cand_name: &str) -> Option<Vec<Vec<Constraint>>> {
        let resolved_name = self.ctx.resolve_op_name(cand_name).to_string();
        let cand_def = self.module.op_defs.get(&resolved_name)?;
        // Part of #3296: TIR re-enabled for candidate branch extraction.
        let tir_program = self.tir_parity.as_ref().and_then(|tir| {
            tir.make_tir_program_for_selected_leaf_eval_name(cand_name, &resolved_name)
        });
        let branches = extract_init_constraints(
            &self.ctx,
            &cand_def.body,
            &self.module.vars,
            tir_program.as_ref(),
        )?;
        if !find_unconstrained_vars(&self.module.vars, &branches).is_empty() {
            return None;
        }
        Some(branches)
    }

    /// Build a candidate-remainder filter expression for fallback type-predicate enumeration.
    ///
    /// Given the full init predicate body and a candidate type predicate name,
    /// extracts the conjunction remainder (the portion of Init not covered by the
    /// candidate). Returns `None` when the remainder is trivially `TRUE`.
    ///
    /// Part of #3322: extracted from three fallback enumeration paths.
    pub(super) fn candidate_remainder_filter_expr(
        &self,
        pred_body: &tla_core::Spanned<tla_core::ast::Expr>,
        cand_name: &str,
    ) -> Option<tla_core::Spanned<tla_core::ast::Expr>> {
        let filter_expr = extract_conjunction_remainder(&self.ctx, pred_body, cand_name)
            .unwrap_or_else(|| pred_body.clone());
        if matches!(filter_expr.node, tla_core::ast::Expr::Bool(true)) {
            None
        } else {
            Some(filter_expr)
        }
    }
}
