// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! ay ALL-SAT Init state enumeration entry point.

use std::collections::HashMap;
use std::sync::Arc;

use tla_core::ast::Expr;
use tla_core::Spanned;

use crate::eval::{eval_entry, EvalCtx};
use crate::state::State;
use crate::symbolic_explore::AYSolveDecisionProfileEvidence;

use super::value_convert::{collect_referenced_zero_arg_ops, try_value_to_ay_spanned_expr};
use super::{AYEnumConfig, AYEnumError, AYEnumResult, VarInfo, VarSort};

#[cfg(feature = "ay")]
use super::model::{blocking_var_names, model_to_state};
#[cfg(feature = "ay")]
use super::type_inference::infer_var_types;

/// Convert VarSort to TlaSort for ay translator
#[cfg(feature = "ay")]
fn varsort_to_tlasort(sort: &VarSort) -> Result<tla_ay::TlaSort, String> {
    use tla_ay::TlaSort;
    match sort {
        VarSort::Bool => Ok(TlaSort::Bool),
        VarSort::Int => Ok(TlaSort::Int),
        VarSort::String { .. } => Ok(TlaSort::String),
        VarSort::Function { .. } => Err("nested function types not supported in tuple".to_string()),
        VarSort::Tuple { element_sorts } => {
            let tla_sorts: Vec<TlaSort> = element_sorts
                .iter()
                .map(varsort_to_tlasort)
                .collect::<Result<_, _>>()?;
            Ok(TlaSort::Tuple {
                element_sorts: tla_sorts,
            })
        }
        VarSort::Heterogeneous { reason } => Err(format!("heterogeneous set: {}", reason)),
    }
}

#[cfg(feature = "ay")]
pub(crate) fn ensure_init_enumeration_sat_profile_accepted(
    solver_profile: &AYSolveDecisionProfileEvidence,
) -> AYEnumResult<()> {
    if solver_profile.accepts_model_for_tla_boundary() {
        return Ok(());
    }

    Err(AYEnumError::SolverFailed(
        "AY SAT result rejected by consumer boundary during Init enumeration".to_string(),
    ))
}

/// Enumerate Init states using ay SMT solver.
///
/// This is the main entry point for ay-based Init enumeration.
/// It translates the Init predicate to ay constraints and uses
/// ALL-SAT with blocking clauses to enumerate all satisfying models.
///
/// # Arguments
/// * `ctx` - Evaluation context with operator definitions
/// * `init_expr` - The Init predicate expression
/// * `vars` - State variables to enumerate
/// * `var_types` - Optional type information for variables
/// * `config` - Enumeration configuration
///
/// # Returns
/// Vector of all States satisfying Init, or error
#[cfg(feature = "ay")]
pub(crate) fn enumerate_init_states_ay(
    ctx: &EvalCtx,
    init_expr: &Spanned<Expr>,
    vars: &[Arc<str>],
    var_types: Option<&HashMap<String, VarInfo>>,
    config: &AYEnumConfig,
) -> AYEnumResult<Vec<State>> {
    use tla_ay::{AYTranslator, SolveResult, TlaSort};

    if config.debug {
        eprintln!(
            "[ay-enum] Starting ay-based Init enumeration for {} variables",
            vars.len()
        );
    }

    // Create translator.
    //
    // native-seq decision: this lane deliberately stays on the bounded array+len
    // encoder (`AYTranslator::new()`), NOT the native unbounded `Sort::Seq` path
    // (`new_with_seq`). Two reasons:
    //   1. This is ALL-SAT Init enumeration (a blocking-clause loop that must
    //      enumerate *every* Init state). Native unbounded sequences make that
    //      ill-posed: a sequence-typed state var ranges over an infinite domain
    //      (e.g. `s \in Seq(S)` has infinitely many models), so ALL-SAT would not
    //      terminate — it would spin to `max_solutions` and error. Explicit-state
    //      enumeration fundamentally needs a finite domain bound, which the
    //      `max_len` array encoding supplies and native `seq.len` does not.
    //   2. This path declares no sequence state vars today (`VarSort` has no
    //      `Sequence` variant); seq-bearing Init is handled faithfully by the
    //      explicit brute-force fallback. Keeping `new()` preserves that.
    // Native unbounded sequences are the right tool for SYMBOLIC property checking
    // over a *single* symbolic sequence (`new_with_seq`, opt-in), not for
    // exhaustive state enumeration.
    let mut translator = AYTranslator::new();

    // Part of #522/#515: Pass constant definitions to translator, pre-evaluating where safe.
    // We only attempt pre-evaluation for constants referenced by Init,
    // since evaluating every constant can be expensive (and can materialize huge sets).
    let mut constant_defs: HashMap<String, Spanned<Expr>> = HashMap::new();
    for (name, def) in ctx.shared().ops.iter() {
        if !def.params.is_empty() {
            continue; // Only zero-arg operators (constants)
        }
        constant_defs.insert(name.clone(), def.body.clone());
    }

    let mut referenced_constants = std::collections::BTreeSet::new();
    collect_referenced_zero_arg_ops(init_expr, ctx, &mut referenced_constants);

    let mut pre_eval_count = 0;
    let mut fallback_count = 0;
    for name in referenced_constants {
        let Some(def) = ctx.shared().ops.get(&name) else {
            continue;
        };

        let evaluated_expr = match eval_entry(ctx, &def.body) {
            Ok(value) => match try_value_to_ay_spanned_expr(&value, def.body.span) {
                Some(expr) => {
                    if config.debug {
                        eprintln!(
                            "[ay-enum] Pre-evaluated constant '{}' to ay-safe Expr",
                            name
                        );
                    }
                    pre_eval_count += 1;
                    expr
                }
                None => {
                    if config.debug {
                        eprintln!(
                            "[ay-enum] Constant '{}' not representable for ay, using raw expr",
                            name
                        );
                    }
                    fallback_count += 1;
                    def.body.clone()
                }
            },
            Err(e) => {
                if config.debug {
                    eprintln!(
                        "[ay-enum] Constant '{}' eval failed ({}), using raw expr",
                        name, e
                    );
                }
                fallback_count += 1;
                def.body.clone()
            }
        };
        constant_defs.insert(name, evaluated_expr);
    }

    if config.debug && (pre_eval_count > 0 || fallback_count > 0) {
        eprintln!(
            "[ay-enum] Constants: {} pre-evaluated, {} fallback",
            pre_eval_count, fallback_count
        );
    }

    // Step 1: Declare variables
    // If we have type info, use it. Otherwise, infer from Init predicate.
    // Pass constant_defs to resolve Ident references during type inference.
    // Must happen BEFORE we move constant_defs to the translator.
    let var_infos = match var_types {
        Some(types) => types.clone(),
        None => infer_var_types(init_expr, vars, &constant_defs)?,
    };

    // Now pass constant_defs to the translator for Ident resolution.
    translator.set_constant_defs(constant_defs);

    for (name, info) in &var_infos {
        match &info.sort {
            VarSort::Bool => {
                translator.declare_var(name, TlaSort::Bool).map_err(|e| {
                    AYEnumError::TranslationFailed(format!("declare {} failed: {}", name, e))
                })?;
            }
            VarSort::Int => {
                translator.declare_var(name, TlaSort::Int).map_err(|e| {
                    AYEnumError::TranslationFailed(format!("declare {} failed: {}", name, e))
                })?;
            }
            VarSort::String { .. } => {
                translator.declare_var(name, TlaSort::String).map_err(|e| {
                    AYEnumError::TranslationFailed(format!("declare {} failed: {}", name, e))
                })?;
            }
            VarSort::Function { domain_keys, range } => {
                let range_sort = match range.as_ref() {
                    VarSort::Bool => TlaSort::Bool,
                    VarSort::Int => TlaSort::Int,
                    VarSort::String { .. } => TlaSort::String,
                    VarSort::Function { .. }
                    | VarSort::Tuple { .. }
                    | VarSort::Heterogeneous { .. } => {
                        return Err(AYEnumError::UnsupportedVarType {
                            var: name.clone(),
                            reason: "nested function/tuple/heterogeneous types not supported"
                                .to_string(),
                        });
                    }
                };
                translator
                    .declare_func_var(name, domain_keys.clone(), range_sort)
                    .map_err(|e| {
                        AYEnumError::TranslationFailed(format!(
                            "declare func {} failed: {}",
                            name, e
                        ))
                    })?;
            }
            VarSort::Tuple { element_sorts } => {
                let tla_sorts: Vec<TlaSort> = element_sorts
                    .iter()
                    .map(|s| varsort_to_tlasort(s))
                    .collect::<Result<_, _>>()
                    .map_err(|reason| AYEnumError::UnsupportedVarType {
                        var: name.clone(),
                        reason,
                    })?;
                translator.declare_tuple_var(name, tla_sorts).map_err(|e| {
                    AYEnumError::TranslationFailed(format!("declare tuple {} failed: {}", name, e))
                })?;
            }
            // Part of #523: Heterogeneous sets cannot be represented in ay
            // Return error to force fallback to brute-force enumeration
            VarSort::Heterogeneous { reason } => {
                return Err(AYEnumError::UnsupportedVarType {
                    var: name.clone(),
                    reason: format!("heterogeneous set membership: {}", reason),
                });
            }
        }
    }

    // Step 2: Translate Init predicate to ay formula
    let init_term = translator
        .translate_bool(init_expr)
        .map_err(|e| AYEnumError::TranslationFailed(format!("Init translation failed: {}", e)))?;

    translator.assert(init_term);

    // Part of #2826: Install solve timeout before the enumeration loop.
    if let Some(timeout) = config.solve_timeout {
        translator.set_timeout(Some(timeout));
    }

    // Step 3: Enumerate all solutions using blocking clauses
    let mut states = Vec::new();
    let mut solution_count = 0;

    loop {
        if solution_count >= config.max_solutions {
            return Err(AYEnumError::MaxSolutionsExceeded(config.max_solutions));
        }

        // Part of #2826/#4449: Use typed solve details so SAT model consumers
        // share AY's decision-profile acceptance boundary.
        let (sat_result, summary) = translator
            .try_check_sat_with_decision_profile_summary()
            .map_err(|e| AYEnumError::SolverFailed(format!("{}", e)))?;
        let solver_profile = AYSolveDecisionProfileEvidence::from_summary("TLA", Some(&summary));

        match sat_result {
            SolveResult::Sat => {
                ensure_init_enumeration_sat_profile_accepted(&solver_profile)?;

                // Part of #2826: Use try_get_model for typed model errors.
                let model = translator
                    .try_get_model()
                    .map_err(|e| AYEnumError::InvalidModel(format!("{}", e)))?;

                // Get string reverse map for converting interned IDs back to strings
                let string_reverse_map = translator.get_string_reverse_map();

                let state = model_to_state(&model, &var_infos, &string_reverse_map)?;

                if config.debug && solution_count < 10 {
                    eprintln!("[ay-enum] Solution {}: {:?}", solution_count, state);
                }

                // Add a translator-owned blocking clause to exclude this solution.
                let blocking_names = blocking_var_names(&var_infos)?;
                translator
                    .assert_model_blocking_clause(&model, blocking_names)
                    .map_err(|e| {
                        AYEnumError::TranslationFailed(format!(
                            "model blocking clause failed: {}",
                            e
                        ))
                    })?;

                states.push(state);
                solution_count += 1;
            }
            SolveResult::Unsat(_) => {
                // No more solutions
                break;
            }
            SolveResult::Unknown => {
                // Part of #2826: Distinguish timeout from other unknown reasons.
                let is_timeout = translator
                    .last_unknown_reason()
                    .map_or(false, |r| matches!(r, tla_ay::UnknownReason::Timeout));
                if is_timeout {
                    return Err(AYEnumError::SolverTimeout);
                }
                return Err(AYEnumError::SolverUnknown);
            }
            _ => {
                return Err(AYEnumError::SolverUnknown);
            }
        }
    }

    if config.debug {
        eprintln!("[ay-enum] Found {} Init states", states.len());
    }

    Ok(states)
}
